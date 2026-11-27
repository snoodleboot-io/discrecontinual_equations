"""The Butcher tableau of an embedded explicit Runge-Kutta pair, in exact rationals.

An explicit Runge-Kutta step of ``s`` stages evaluates the field at
``k_i = f(t + c_i h, y + h sum_{j<i} a_ij k_j)`` and advances with
``y + h sum_i b_i k_i``; an embedded pair carries a second weight row ``b_hat``
on the same stages, and ``h sum_i (b_i - b_hat_i) k_i`` estimates the local
error. The numbers ``a``, ``b``, ``b_hat`` and ``c`` are the whole method, so a
single wrong digit changes its order (DEQ-137, DEQ-138, DEQ-139).

The coefficients are kept here as :class:`fractions.Fraction`, as printed in
their sources, so that a test can check the order conditions - one equation per
rooted tree, Butcher's theorem - in exact arithmetic rather than to a float
tolerance; the solvers read the same numbers rounded once to ``float``.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

Row = tuple[Fraction, ...]


@dataclass(frozen=True)
class ButcherTableau:
    """An embedded explicit Runge-Kutta pair: ``a``, ``c``, ``b`` and ``b_hat``.

    ``b`` is the row the step advances with and ``b_hat`` the embedded row
    whose difference from it is the error estimate; ``order`` and
    ``embedded_order`` are their orders. Advancing with the higher-order row,
    as all four pairs in this package do, is "local extrapolation": the error
    estimate then belongs to the lower-order row, so it is pessimistic for the
    solution actually returned.

    Attributes:
        name: The pair's usual name, e.g. ``"Dormand-Prince 5(4)"``.
        a: The strictly lower-triangular stage matrix, one row per stage.
        c: The nodes; for every tableau here ``c_i = sum_j a_ij``.
        b: The weights the step advances with.
        b_hat: The embedded weights.
        order: The order of ``b``.
        embedded_order: The order of ``b_hat``.

    References:
        Butcher, J. C. (1963). Coefficients for the study of Runge-Kutta
        integration processes. Journal of the Australian Mathematical
        Society 3(2):185-201. DOI: 10.1017/S1446788700027932. Eqs. (4)-(5),
        the process; the introduction, which makes a non-autonomous system
        autonomous by adding ``x`` as a component with ``dx/dx = 1`` - the
        reason ``c_i`` must be ``sum_j a_ij``; and eqs. (27)-(37), one
        condition ``sum_i b_i Phi_i(t) = 1/gamma(t)`` per rooted tree ``t``.
        See ``docs/references.md`` (Butcher1963).
    """

    name: str
    a: tuple[Row, ...]
    c: Row
    b: Row
    b_hat: Row
    order: int
    embedded_order: int

    @property
    def stages(self) -> int:
        """The number of stages ``s``."""
        return len(self.c)

    @property
    def error_order(self) -> int:
        """The order of the error estimate, the lower of the two orders.

        The local error of that row goes as ``h^(error_order + 1)``, which is
        the exponent a step-size controller inverts.
        """
        return min(self.order, self.embedded_order)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(a, c, b, b_hat)`` as float arrays, each entry rounded once.

        ``a`` is padded to the full ``s x s`` lower-triangular matrix.
        """
        size = self.stages
        a = np.zeros((size, size))
        for i, row in enumerate(self.a):
            a[i, : len(row)] = [float(value) for value in row]
        return (
            a,
            np.array([float(value) for value in self.c]),
            np.array([float(value) for value in self.b]),
            np.array([float(value) for value in self.b_hat]),
        )


def row(text: str) -> Row:
    """A row of exact rationals from whitespace-separated ``p/q`` or integer entries.

    ``row("35/384 0 500/1113")`` is ``(Fraction(35, 384), Fraction(0),
    Fraction(500, 1113))``: the coefficients read as they are printed.
    """
    return tuple(Fraction(entry) for entry in text.split())
