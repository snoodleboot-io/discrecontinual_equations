"""Fold, flip and torus curves of limit cycles in two parameters, and where they meet.

A codimension-one cycle bifurcation found on a one-parameter branch
(:class:`~.cycle_continuation.CycleBifurcation`) persists along a curve in two
parameters. :class:`CycleCurveContinuation` follows it by pseudo-arclength on
the cycle's collocation system - the same discretisation, sparse assembly and
phase condition as :class:`~.cycle_continuation.CycleContinuation` - extended by
the bifurcation condition in *minimally augmented* form (Govaerts; Doedel,
Govaerts and Kuznetsov, "Computation of periodic solution bifurcations in ODEs
using bordered systems", SIAM J. Numer. Anal. 41, 2003): a linear operator
built from the collocation blocks, singular exactly on the curve, is bordered
to a nonsingular one, and the bordered system's corner is the test function.

* **fold** (``fold_of_cycles``, MATCONT's LPC): the Jacobian of the cycle
  system in the states and the period - the one-parameter corrector's matrix -
  is singular. One test function.
* **flip** (``period_doubling``, PD): the interval blocks with the
  antiperiodic condition ``v_N + v_0 = 0`` are singular, i.e. the discrete
  monodromy has the multiplier ``-1``. One test function.
* **torus** (``neimark_sacker``, NS): on the doubled mesh the blocks with
  ``v_2N - 2 kappa v_N + v_0 = 0`` are singular, i.e. ``M^2 - 2 kappa M + I`` is,
  which holds exactly when ``e^(+-i theta)`` with ``cos theta = kappa`` are
  multipliers. Its null space is two-dimensional there, so it is bordered by
  two vectors each way and two entries of the ``2 x 2`` corner are the test
  functions, with ``kappa`` one more unknown.

Because each operator is built from the very blocks whose transfer matrices
give the discrete monodromy, the curve is where *that* monodromy has the
critical multiplier, to rounding.

The gradient of a corner entry ``g = G_kl`` is ``-W_k^T (dL) V_l`` with ``V`` and
``W`` the bordered system's right and left solutions. Its state and period
components come from one more assembly of the blocks, at the states displaced
along ``V_l``, since ``d/dz (w^T L(z) v) = d/de [J(z + e v)]^T w`` for ``L``
built from the Jacobian ``J`` (second derivatives commute); its parameter
components by a forward difference in each parameter.

Along a curve the codimension-two points are located where a test function
changes sign between adjacent points, as on a cycle branch: a second critical
multiplier (``fold_flip``, ``fold_neimark_sacker``, ``flip_neimark_sacker``,
``double_neimark_sacker``, ``strong_resonance_1_2`` on a flip curve), the
strong resonances on a torus curve as ``kappa`` passes ``1, -1, -1/2, 0``
(``strong_resonance_1_1`` to ``_1_4``), and the normal form coefficient of the
curve itself vanishing (``cusp_of_cycles``, ``generalized_period_doubling``,
``chenciner``; see :mod:`.cycle_normal_form`).

Limits:

- Only the one-point-per-interval schemes (Hermite-Simpson, trapezoidal) are
  supported; a Gauss-Legendre ``collocation`` is refused with ``ValueError``.
- A codimension-two point is *located*: a test function changing sign
  between two adjacent curve points is driven to zero by the secant method on
  the curve between them (eight trials at most, back to the linear estimate
  if a trial fails to correct). Its own normal form (De Witte et al.'s
  coefficients, which decide the unfolding) is not computed.
- The test functions are read off the discrete monodromy's multipliers and
  the coefficients off the discrete cycle, so the located points carry the
  discretisation's error, and two sign changes within one step cancel and
  are not seen.
- The corner's gradient is differenced, not exact: its state and period
  components from the blocks re-assembled at the states displaced along
  ``V_l``, its parameter components by forward differences, both with steps
  of ``1e-7`` relative to the quantity's size.
- ``M^2 - 2 kappa M + I`` is singular also for a real pair ``mu, 1/mu``,
  ``kappa = (mu + 1/mu) / 2`` outside ``[-1, 1]``: past a 1:1 or 1:2
  resonance a torus curve continues as a curve of neutral saddle cycles,
  where no ``d`` is computed and no Chenciner point is looked for.

References:
    - [Doedel2003] Doedel, E. J., Govaerts, W. and Kuznetsov, Yu. A.
      "Computation of periodic solution bifurcations in ODEs using bordered
      systems", SIAM J. Numer. Anal. 41(2), 401-435, 2003.
      DOI 10.1137/S0036142902400779. The bordered (minimally augmented)
      defining systems for fold, flip and torus curves of cycles.
    - [DeWitte2013] De Witte, V., Della Rossa, F., Govaerts, W. and
      Kuznetsov, Yu. A. "Numerical periodic normalization for codim 2
      bifurcations of limit cycles: computational formulas, numerical
      implementation, and examples", SIAM J. Appl. Dyn. Syst. 12(2),
      722-788, 2013. DOI 10.1137/120874904; preprint arXiv:1111.4445.
      Section 2, as numbered in the preprint: the codimension-two points of
      cycles - cusp (2.1.1), generalised period-doubling (2.1.2), Chenciner
      (2.2.1), strong resonances 1:1 to 1:4 (2.2.2-2.2.5), fold-flip (2.2.6)
      and the rest - which the test functions here locate.
    - [Kuznetsov2005] Kuznetsov, Yu. A., Govaerts, W., Doedel, E. J. and
      Dhooge, A. "Numerical periodic normalization for codim 1 bifurcations
      of limit cycles", SIAM J. Numer. Anal. 43(4), 1407-1435, 2005.
      DOI 10.1137/040611306. The coefficients whose zeros end a curve's
      nondegenerate stretch.
"""

from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, hstack, vstack
from scipy.sparse.linalg import splu

from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CyclePoint,
    _product_multipliers,
    _torus_excess,
    _transfer_matrices,
)
from discrecontinual_equations.continuation.cycle_normal_form import CycleNormalForm
from discrecontinual_equations.differential_equation import DifferentialEquation

_KINDS = ("fold_of_cycles", "period_doubling", "neimark_sacker")
_TOLERANCE = 1.0e-9
_ITERATIONS = 12
_DIFFERENCE = 1.0e-7
_DISPLACEMENT = 1.0e-7
# Imaginary parts below this are a real multiplier's rounding, not a pair.
_IMAGINARY = 1.0e-9
# The secant that pins a codimension-two point down between two curve points.
_LOCATE_ITERATIONS = 8
_LOCATE_TOLERANCE = 1.0e-8
# kappa at the four strong resonances: e^(i theta) with theta = 0, pi, 2pi/3, pi/2.
_RESONANCES = (
    ("strong_resonance_1_1", 1.0),
    ("strong_resonance_1_2", -1.0),
    ("strong_resonance_1_3", -0.5),
    ("strong_resonance_1_4", 0.0),
)
_COEFFICIENT_ZEROS = {
    "fold_of_cycles": "cusp_of_cycles",
    "period_doubling": "generalized_period_doubling",
    "neimark_sacker": "chenciner",
}


class CycleCurvePoint:
    """One cycle on a codimension-one curve: both parameters, orbit, multipliers.

    ``kappa`` is ``cos theta`` of the critical pair on a torus curve and
    ``None`` elsewhere; ``coefficient`` the normal form coefficient of the
    curve's bifurcation there (``b``, ``c``, or the complex ``d``), ``None``
    when not computed. ``tests`` holds each codimension-two test function's
    value at the point, keyed by the point it detects.

    Args:
        parameters: The two parameters' values.
        states: The cycle's node states.
        period: Its period.
        multipliers: Its Floquet multipliers.
        kappa: ``cos theta`` of the critical pair, torus curves only.
        coefficient: The curve's normal form coefficient, or ``None``.

    References:
        - [Doedel2003] Doedel, E. J., Govaerts, W. and Kuznetsov, Yu. A.
          "Computation of periodic solution bifurcations in ODEs using
          bordered systems", SIAM J. Numer. Anal. 41(2), 401-435, 2003.
          DOI 10.1137/S0036142902400779.
        - [Kuznetsov2005] Kuznetsov, Govaerts, Doedel and Dhooge, SIAM J.
          Numer. Anal. 43(4), 1407-1435, 2005. DOI 10.1137/040611306.
          ``b``, ``c`` and ``d``.
    """

    __slots__ = [
        "coefficient",
        "kappa",
        "multipliers",
        "parameters",
        "period",
        "states",
        "tests",
    ]

    def __init__(  # noqa: PLR0913 (a record)
        self,
        parameters: tuple[float, float],
        states: np.ndarray,
        period: float,
        multipliers: np.ndarray,
        kappa: float | None,
        coefficient: complex | None,
    ) -> None:
        self.parameters = parameters
        self.states = states
        self.period = period
        self.multipliers = multipliers
        self.kappa = kappa
        self.coefficient = coefficient
        self.tests: dict[str, float] = {}


class CycleCodim2Point:
    """A codimension-two point located on a curve, between two of its points.

    Args:
        kind: Which point: ``cusp_of_cycles``, ``generalized_period_doubling``,
            ``chenciner``, ``fold_flip``, ``fold_neimark_sacker``,
            ``flip_neimark_sacker``, ``double_neimark_sacker`` or
            ``strong_resonance_1_1`` to ``strong_resonance_1_4``.
        parameters: Where it lies, interpolated linearly between the two
            curve points whose test functions differ in sign.

    References:
        - [DeWitte2013] De Witte, V., Della Rossa, F., Govaerts, W. and
          Kuznetsov, Yu. A. "Numerical periodic normalization for codim 2
          bifurcations of limit cycles: computational formulas, numerical
          implementation, and examples", SIAM J. Appl. Dyn. Syst. 12(2),
          722-788, 2013. DOI 10.1137/120874904; preprint arXiv:1111.4445. Section 2.
    """

    __slots__ = ["kind", "parameters"]

    def __init__(self, kind: str, parameters: tuple[float, float]) -> None:
        self.kind = kind
        self.parameters = parameters


class CycleCurve:
    """A traced fold, flip or torus curve of cycles and its codimension-two points.

    Args:
        kind: ``fold_of_cycles``, ``period_doubling`` or ``neimark_sacker``.
        points: The curve's points in the order traced; empty if the start
            could not be corrected onto the curve.
        special: The codimension-two points found between adjacent points.

    References:
        - [Doedel2003] Doedel, E. J., Govaerts, W. and Kuznetsov, Yu. A.
          "Computation of periodic solution bifurcations in ODEs using
          bordered systems", SIAM J. Numer. Anal. 41(2), 401-435, 2003.
          DOI 10.1137/S0036142902400779.
    """

    __slots__ = ["kind", "points", "special"]

    def __init__(
        self,
        kind: str,
        points: list[CycleCurvePoint],
        special: list[CycleCodim2Point],
    ) -> None:
        self.kind = kind
        self.points = points
        self.special = special


class _Bordered:
    """The bordered operator at one point: values, solutions, factorisation."""

    __slots__ = ["corner", "left", "right"]

    def __init__(self, corner: np.ndarray, right: np.ndarray, left: np.ndarray) -> None:
        self.corner = corner
        self.right = right
        self.left = left


def _bordered(matrix: csc_matrix, right: np.ndarray, left: np.ndarray) -> _Bordered:
    """Solve ``[[L, P], [Q^T, 0]] [V; G] = [0; I]`` and its transpose."""
    size, rank = matrix.shape[0], right.shape[1]
    full = vstack(
        [
            hstack([matrix, csc_matrix(left)]),
            hstack([csc_matrix(right.T), csc_matrix((rank, rank))]),
        ],
    ).tocsc()
    factor = splu(full)
    rhs = np.zeros((size + rank, rank))
    rhs[size:] = np.eye(rank)
    forward = factor.solve(rhs)
    backward = factor.solve(rhs, trans="T")
    return _Bordered(forward[size:], forward[:size], backward[:size])


def _orthonormal(vectors: np.ndarray) -> np.ndarray:
    basis, _ = np.linalg.qr(vectors)
    return basis


class CycleCurveContinuation:
    """Pseudo-arclength continuation of a cycle bifurcation in two parameters.

    ``parameter_indices`` are the two parameters, the first the one the start
    point was traced in. The cycle is discretised as
    :class:`CycleContinuation` with the same ``intervals``, phase and
    ``collocation`` (a one-point-per-interval scheme). The unknowns are the
    cycle's states, its period, both parameters and, on a torus curve,
    ``kappa``; the equations are the cycle's collocation system, its phase
    condition, the bifurcation's test functions (the bordered operator's
    corner entries) and the arclength condition. The step is a fixed
    pseudo-arclength in AUTO's ``L2`` norm of the orbit and both parameters.

    Args:
        equation: The system; its first two parameters by
            ``parameter_indices`` are the ones the curve moves.
        parameter_indices: The two active parameters, the first the one the
            start point was traced in.
        intervals: Mesh intervals, as :class:`CycleContinuation`.
        phase_index: The coordinate the point phase condition pins.
        phase_value: The value it is pinned to.
        collocation: ``"hermite_simpson"`` or ``"trapezoidal"``.
        normal_forms: Compute the curve's normal form coefficient at every
            point (needed for the cusp, generalised period-doubling and
            Chenciner points).

    Raises:
        ValueError: If ``collocation`` is not a one-point-per-interval
            scheme, or :class:`CycleContinuation` refuses its arguments.

    References:
        - [Doedel2003] Doedel, E. J., Govaerts, W. and Kuznetsov, Yu. A.
          "Computation of periodic solution bifurcations in ODEs using
          bordered systems", SIAM J. Numer. Anal. 41(2), 401-435, 2003.
          DOI 10.1137/S0036142902400779.
        - [DeWitte2013] De Witte, V., Della Rossa, F., Govaerts, W. and
          Kuznetsov, Yu. A. "Numerical periodic normalization for codim 2
          bifurcations of limit cycles: computational formulas, numerical
          implementation, and examples", SIAM J. Appl. Dyn. Syst. 12(2),
          722-788, 2013. DOI 10.1137/120874904; preprint arXiv:1111.4445.
    """

    __slots__ = [
        "_borders",
        "_cycle",
        "_dimension",
        "_equation",
        "_kind",
        "_nodes",
        "_normal_forms",
        "_pair",
        "_parameters",
        "_rank",
    ]

    def __init__(  # noqa: PLR0913 (the cycle's discretisation, as CycleContinuation)
        self,
        equation: DifferentialEquation,
        parameter_indices: tuple[int, int],
        intervals: int,
        phase_index: int = 0,
        phase_value: float = 0.0,
        *,
        collocation: str = "hermite_simpson",
        normal_forms: bool = True,
    ) -> None:
        self._cycle = CycleContinuation(
            equation,
            parameter_indices[0],
            intervals,
            phase_index,
            phase_value,
            collocation=collocation,
        )
        if self._cycle.collocation_points != 1:
            message = "curves of cycles take a one-point-per-interval scheme"
            raise ValueError(message)
        self._equation = equation
        self._parameters = parameter_indices
        self._normal_forms = normal_forms
        self._kind = ""
        self._rank = 1
        self._dimension = 0
        self._nodes = intervals + 1
        self._borders: tuple[np.ndarray, np.ndarray] | None = None
        self._pair: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (1, 1))

    # -- layout ----------------------------------------------------------------

    @property
    def _cycle_size(self) -> int:
        """States, period and first parameter: the cycle continuation's unknowns."""
        return self._nodes * self._dimension + 2

    def _set_second(self, value: float) -> None:
        parameter = self._equation.derivative.parameters[self._parameters[1]]
        parameter.value = float(value)

    # -- the cycle system ------------------------------------------------------

    def _jacobian(self, unknowns: np.ndarray) -> tuple[csc_matrix, np.ndarray]:
        """The cycle system's Jacobian in all its unknowns, and its interval blocks."""
        cycle = self._cycle
        size = self._cycle_size
        self._set_second(unknowns[size])
        blocks = cycle._blocks  # noqa: SLF001
        rows, columns, values = blocks(unknowns[:size], self._nodes)
        assembled = cycle._assembled  # noqa: SLF001
        if assembled is None:  # pragma: no cover (``_blocks`` always sets it)
            message = "the collocation blocks were not assembled"
            raise RuntimeError(message)
        residual = cycle._cycle_residual(unknowns[:size], self._nodes)  # noqa: SLF001
        step = _DIFFERENCE * max(1.0, abs(float(unknowns[size])))
        self._set_second(unknowns[size] + step)
        moved = cycle._cycle_residual(unknowns[:size], self._nodes)  # noqa: SLF001
        self._set_second(unknowns[size])
        second = (moved - residual) / step
        matrix = coo_matrix(
            (values, (rows, columns)),
            shape=(residual.size, size),
        ).tocsc()
        return hstack([matrix, csc_matrix(second[:, None])]).tocsc(), assembled[1]

    def _residual(self, unknowns: np.ndarray) -> np.ndarray:
        size = self._cycle_size
        self._set_second(unknowns[size])
        return self._cycle._cycle_residual(unknowns[:size], self._nodes)  # noqa: SLF001

    # -- the singular operators ------------------------------------------------

    def _operator(
        self,
        jacobian: csc_matrix,
        blocks: np.ndarray,
        kappa: float,
    ) -> csc_matrix:
        """``L``: singular exactly on the curve (see the module docstring)."""
        dimension = self._dimension
        if self._kind == "fold_of_cycles":
            # The cycle Jacobian in the states and the period.
            return jacobian[:, : self._cycle_size - 1].tocsc()
        intervals = blocks.shape[0]
        copies = 1 if self._kind == "period_doubling" else 2
        size = (copies * intervals + 1) * dimension
        rows, columns, values = [], [], []
        span = np.arange(dimension)
        for copy in range(copies):
            for interval in range(intervals):
                start = (copy * intervals + interval) * dimension
                grid_rows = start + span[:, None] + 0 * span[None, :]
                grid_columns = start + span[None, :] + 0 * span[:, None]
                rows += [grid_rows.ravel(), grid_rows.ravel()]
                columns += [grid_columns.ravel(), (grid_columns + dimension).ravel()]
                values += [
                    blocks[interval, :, :dimension].ravel(),
                    blocks[interval, :, dimension:].ravel(),
                ]
        last = copies * intervals * dimension
        boundary = last + span
        if copies == 1:  # v_N + v_0 = 0
            rows += [boundary, boundary]
            columns += [last + span, span]
            values += [np.ones(dimension), np.ones(dimension)]
        else:  # v_2N - 2 kappa v_N + v_0 = 0
            rows += [boundary, boundary, boundary]
            columns += [last + span, intervals * dimension + span, span]
            values += [
                np.ones(dimension),
                -2.0 * kappa * np.ones(dimension),
                np.ones(dimension),
            ]
        return coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(size, size),
        ).tocsc()

    def _copies(self, vector: np.ndarray) -> list[tuple[slice, slice]]:
        """Where each copy of the cycle sits in ``L``'s rows and columns."""
        dimension = self._dimension
        intervals = self._nodes - 1
        if self._kind == "fold_of_cycles":
            return [(slice(None), slice(0, vector.size))]
        copies = 1 if self._kind == "period_doubling" else 2
        return [
            (
                slice(copy * intervals * dimension, (copy + 1) * intervals * dimension),
                slice(
                    copy * intervals * dimension,
                    ((copy + 1) * intervals + 1) * dimension,
                ),
            )
            for copy in range(copies)
        ]

    # -- the extended system ---------------------------------------------------

    def _tests(
        self, unknowns: np.ndarray, kappa: float
    ) -> tuple[np.ndarray, _Bordered]:
        jacobian, blocks = self._jacobian(unknowns)
        operator = self._operator(jacobian, blocks, kappa)
        if self._borders is None:  # pragma: no cover (set before any solve)
            message = "bordering vectors are not set"
            raise RuntimeError(message)
        solved = _bordered(operator, self._borders[0], self._borders[1])
        return self._selected(solved.corner), solved

    def _selected(self, corner: np.ndarray) -> np.ndarray:
        if self._rank == 1:
            return np.array([corner[0, 0]])
        return np.array([corner[index] for index in self._pair])

    def _gradient(
        self,
        unknowns: np.ndarray,
        kappa: float,
        jacobian: csc_matrix,
        solved: _Bordered,
    ) -> np.ndarray:
        """Gradient of every corner entry in ``(states, period, p1, p2, kappa)``.

        Shape ``(rank, rank, unknowns + 1)``: the last column is ``kappa``'s.
        """
        size = self._cycle_size
        rank = self._rank
        total = unknowns.size + 1
        gradient = np.zeros((rank, rank, total))
        rows = jacobian.shape[0]
        scale = 1.0 + float(np.max(np.abs(unknowns[: size - 2])))
        for column in range(rank):
            right = solved.right[:, column]
            for row_slice, column_slice in self._copies(right):
                displacement = np.zeros(unknowns.size)
                piece = right[column_slice]
                displacement[: piece.size] = piece
                step = _DISPLACEMENT * scale / max(float(np.max(np.abs(piece))), 1e-300)
                moved, _ = self._jacobian(unknowns + step * displacement)
                change = (moved - jacobian) / step
                for k in range(rank):
                    weights = np.zeros(rows)
                    left = solved.left[row_slice, k]
                    weights[: left.size] = left
                    # Only the state and period columns: the parameter columns
                    # are differences themselves and are taken below instead.
                    gradient[k, column, : size - 1] -= change.T.dot(weights)[: size - 1]
        for index in (size - 1, size):
            step = _DIFFERENCE * max(1.0, abs(float(unknowns[index])))
            shifted = unknowns.copy()
            shifted[index] += step
            moved_corner = self._tests_corner(shifted, kappa)
            base_corner = solved.corner
            gradient[:, :, index] = (moved_corner - base_corner) / step
        if self._kind == "neimark_sacker":
            dimension = self._dimension
            intervals = self._nodes - 1
            boundary = slice(2 * intervals * dimension, (2 * intervals + 1) * dimension)
            middle = slice(intervals * dimension, (intervals + 1) * dimension)
            for k in range(rank):
                for column in range(rank):
                    gradient[k, column, -1] = 2.0 * float(
                        solved.left[boundary, k] @ solved.right[middle, column],
                    )
        return gradient

    def _tests_corner(self, unknowns: np.ndarray, kappa: float) -> np.ndarray:
        _, solved = self._tests(unknowns, kappa)
        return solved.corner

    def _extended(
        self,
        unknowns: np.ndarray,
        kappa: float,
    ) -> tuple[np.ndarray, csc_matrix, _Bordered]:
        """``[F; g]`` and its Jacobian in ``(states, period, p1, p2[, kappa])``."""
        jacobian, blocks = self._jacobian(unknowns)
        operator = self._operator(jacobian, blocks, kappa)
        if self._borders is None:  # pragma: no cover (set before any solve)
            message = "bordering vectors are not set"
            raise RuntimeError(message)
        solved = _bordered(operator, self._borders[0], self._borders[1])
        gradient = self._gradient(unknowns, kappa, jacobian, solved)
        residual = self._residual(unknowns)
        if self._rank == 1:
            values = np.array([solved.corner[0, 0]])
            rows = gradient[0, 0][None, : unknowns.size]
            columns = jacobian
        else:
            values = np.array([solved.corner[index] for index in self._pair])
            rows = np.array([gradient[index] for index in self._pair])
            columns = hstack([jacobian, csc_matrix((jacobian.shape[0], 1))]).tocsc()
        matrix = vstack([columns, csc_matrix(rows)]).tocsc()
        return np.concatenate([residual, values]), matrix, solved

    def _pack(self, unknowns: np.ndarray, kappa: float) -> np.ndarray:
        if self._rank == 1:
            return unknowns
        return np.concatenate([unknowns, [kappa]])

    def _unpack(self, packed: np.ndarray) -> tuple[np.ndarray, float]:
        if self._rank == 1:
            return packed, 0.0
        return packed[:-1], float(packed[-1])

    def _newton(
        self,
        packed: np.ndarray,
        row: np.ndarray,
        target: float,
    ) -> np.ndarray | None:
        """Newton on ``[F; g; row . z - target]``; ``None`` if it does not converge."""
        for _ in range(_ITERATIONS):
            unknowns, kappa = self._unpack(packed)
            values, matrix, _ = self._extended(unknowns, kappa)
            constraint = float(row @ packed - target)
            residual = np.concatenate([values, [constraint]])
            if not np.all(np.isfinite(residual)):
                return None
            full = vstack([matrix, csc_matrix(row[None, :])]).tocsc()
            try:
                update = splu(full).solve(-residual)
            except RuntimeError:
                return None
            if not np.all(np.isfinite(update)):
                return None
            packed = packed + update
            if np.linalg.norm(update) < _TOLERANCE * (1.0 + np.linalg.norm(packed)):
                return packed
        return None

    def _refresh_borders(self, packed: np.ndarray) -> None:
        unknowns, kappa = self._unpack(packed)
        _, solved = self._tests(unknowns, kappa)
        self._borders = (_orthonormal(solved.right), _orthonormal(solved.left))

    def _initial_borders(self, unknowns: np.ndarray, kappa: float) -> None:
        jacobian, blocks = self._jacobian(unknowns)
        operator = self._operator(jacobian, blocks, kappa).toarray()
        left, _, right = np.linalg.svd(operator)
        self._borders = (right[-self._rank :].T.copy(), left[:, -self._rank :].copy())
        if self._rank == 2:  # noqa: PLR2004 (the torus's two test functions)
            self._pair = self._best_pair(unknowns, kappa)

    def _best_pair(
        self,
        unknowns: np.ndarray,
        kappa: float,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """The two corner entries whose gradients are the most independent."""
        jacobian, _ = self._jacobian(unknowns)
        _, solved = self._tests(unknowns, kappa)
        gradient = self._gradient(unknowns, kappa, jacobian, solved)
        entries = [(0, 0), (0, 1), (1, 0), (1, 1)]
        best, best_score = ((0, 0), (1, 1)), -1.0
        for first, second in combinations(entries, 2):
            pair = np.array([gradient[first], gradient[second]])
            norms = np.linalg.norm(pair, axis=1)
            if np.any(norms == 0.0):
                continue
            singular = np.linalg.svd(pair / norms[:, None], compute_uv=False)
            if singular[-1] > best_score:
                best, best_score = (first, second), float(singular[-1])
        return best

    # -- tracing -----------------------------------------------------------------

    def _weights(self, size: int) -> np.ndarray:
        """AUTO's L2 arclength: the orbit by the trapezoidal rule, both parameters."""
        intervals = self._nodes - 1
        node = np.full(self._nodes, 1.0 / intervals)
        node[[0, -1]] *= 0.5
        weights = np.concatenate(
            [np.repeat(node, self._dimension), [0.0, 1.0, 1.0]],
        )
        if size > weights.size:
            weights = np.concatenate([weights, np.zeros(size - weights.size)])
        return weights

    def _tangent(self, packed: np.ndarray, previous: np.ndarray) -> np.ndarray:
        unknowns, kappa = self._unpack(packed)
        _, matrix, _ = self._extended(unknowns, kappa)
        weights = self._weights(packed.size)
        border = weights * previous
        full = vstack([matrix, csc_matrix(border[None, :])]).tocsc()
        rhs = np.zeros(packed.size)
        rhs[-1] = 1.0
        tangent = splu(full).solve(rhs)
        tangent /= np.sqrt(tangent @ (weights * tangent))
        if tangent @ border < 0.0:
            tangent = -tangent
        return tangent

    def trace(
        self,
        start: CyclePoint,
        kind: str,
        arclength: float,
        steps: int,
        direction: float = 1.0,
    ) -> CycleCurve:
        """Continue the ``kind`` bifurcation from ``start`` in both parameters.

        ``start`` is a cycle at or near the bifurcation, at the current value of
        the second parameter; it is first corrected onto the curve with the
        second parameter held. ``direction`` is the initial sense of the second
        parameter. The second parameter is handed back at its starting value
        when the trace ends.

        Args:
            start: A cycle near the bifurcation, e.g. the traced point of a
                :meth:`CycleContinuation.trace` nearest a crossing it reported.
            kind: ``"fold_of_cycles"``, ``"period_doubling"`` or
                ``"neimark_sacker"``.
            arclength: The pseudo-arclength step.
            steps: How many steps to take.
            direction: The initial sense of the second parameter, +1 or -1.

        Returns:
            The curve, possibly empty if the start cannot be corrected onto it;
            it ends early where the corrector fails.

        Raises:
            ValueError: If ``kind`` is not one of the three, the second
                parameter has no value, or a torus curve's start has no complex
                multiplier pair.

        References:
            - [Doedel2003] Doedel, E. J., Govaerts, W. and Kuznetsov, Yu. A.
              "Computation of periodic solution bifurcations in ODEs using
              bordered systems", SIAM J. Numer. Anal. 41(2), 401-435, 2003.
              DOI 10.1137/S0036142902400779.
            - [DeWitte2013] De Witte, V., Della Rossa, F., Govaerts, W. and
              Kuznetsov, Yu. A. "Numerical periodic normalization for codim 2
              bifurcations of limit cycles: computational formulas, numerical
              implementation, and examples", SIAM J. Appl. Dyn. Syst. 12(2),
              722-788, 2013. DOI 10.1137/120874904; preprint arXiv:1111.4445.
        """
        if kind not in _KINDS:
            message = f"kind must be one of {list(_KINDS)}, got {kind!r}"
            raise ValueError(message)
        self._kind = kind
        self._rank = 2 if kind == "neimark_sacker" else 1
        states = start.solution.states
        self._dimension = states.shape[1]
        self._nodes = states.shape[0]
        cycle = self._cycle
        cycle._dimension = self._dimension  # noqa: SLF001
        held = self._equation.derivative.parameters[self._parameters[1]].value
        if held is None:
            message = "the second parameter needs a value to start the curve at"
            raise ValueError(message)
        second = float(held)
        unknowns = np.concatenate(
            [states.ravel(), [start.solution.period, start.parameter, second]],
        )
        kappa = self._initial_kappa(start.multipliers)
        self._initial_borders(unknowns, kappa)
        packed = self._pack(unknowns, kappa)
        try:
            return self._traced(packed, second, (arclength, steps, direction))
        finally:
            # The curve moves the second parameter; hand it back as it came.
            self._set_second(second)

    def _traced(
        self,
        packed: np.ndarray,
        second: float,
        extent: tuple[float, int, float],
    ) -> CycleCurve:
        arclength, steps, direction = extent
        kind = self._kind
        hold = np.zeros(packed.size)
        hold[self._cycle_size] = 1.0
        corrected = self._newton(packed, hold, float(second))
        if corrected is None:
            return CycleCurve(kind, [], [])
        self._refresh_borders(corrected)
        tangent = self._tangent(corrected, direction * hold)
        points = [self._point(corrected)]
        special: list[CycleCodim2Point] = []
        packed = corrected
        weights = self._weights(packed.size)
        for _ in range(steps):
            row = weights * tangent
            prediction = packed + arclength * tangent
            following = self._newton(prediction, row, float(row @ packed) + arclength)
            if following is None:
                break
            point = self._point(following)
            special.extend(
                self._locate(name, packed, tangent, arclength, points[-1], point)
                for name in _changed(points[-1], point)
            )
            self._refresh_borders(following)
            tangent = self._tangent(following, tangent)
            packed = following
            points.append(point)
        return CycleCurve(kind, points, special)

    def _locate(  # noqa: PLR0913 (the bracket's two ends and how to walk it)
        self,
        name: str,
        packed: np.ndarray,
        tangent: np.ndarray,
        arclength: float,
        lower: CycleCurvePoint,
        upper: CycleCurvePoint,
    ) -> CycleCodim2Point:
        """Where test ``name`` vanishes between two points, by the secant method.

        Each trial point is corrected onto the curve on the plane at arclength
        ``s`` along the lower point's tangent, so the bracket is a stretch of
        the curve itself. Falls back to the linear estimate when a trial fails.
        """
        weights = self._weights(packed.size)
        row = weights * tangent
        base = float(row @ packed)
        low, high = 0.0, arclength
        f_low, f_high = lower.tests[name], upper.tests[name]
        estimate = _interpolated(name, lower, upper)
        for _ in range(_LOCATE_ITERATIONS):
            if f_high == f_low:
                break
            trial = low - f_low * (high - low) / (f_high - f_low)
            if not low < trial < high:
                trial = 0.5 * (low + high)
            corrected = self._newton(packed + trial * tangent, row, base + trial)
            if corrected is None:
                return estimate
            point = self._point(corrected)
            value = point.tests[name]
            estimate = CycleCodim2Point(name, point.parameters)
            if abs(value) <= _LOCATE_TOLERANCE * max(abs(f_low), abs(f_high)):
                break
            if value * f_low < 0.0:
                high, f_high = trial, value
            else:
                low, f_low = trial, value
        return estimate

    def _initial_kappa(self, multipliers: np.ndarray) -> float:
        if self._kind != "neimark_sacker":
            return 0.0
        complex_pair = [m for m in multipliers if abs(m.imag) > _IMAGINARY]
        if not complex_pair:
            message = "a torus curve starts at a cycle with a complex multiplier pair"
            raise ValueError(message)
        nearest = min(complex_pair, key=lambda m: abs(abs(m) - 1.0))
        return float(np.cos(np.angle(nearest)))

    def _point(self, packed: np.ndarray) -> CycleCurvePoint:
        unknowns, kappa = self._unpack(packed)
        size = self._cycle_size
        dimension = self._dimension
        states = unknowns[: self._nodes * dimension].reshape(self._nodes, dimension)
        period = float(unknowns[size - 2])
        parameters = (float(unknowns[size - 1]), float(unknowns[size]))
        _, blocks = self._jacobian(unknowns)
        multipliers, _ = _product_multipliers(_transfer_matrices(blocks, dimension))
        coefficient = self._coefficient(states, period, parameters, multipliers, kappa)
        point = CycleCurvePoint(
            parameters,
            states.copy(),
            period,
            multipliers,
            kappa if self._kind == "neimark_sacker" else None,
            coefficient,
        )
        point.tests = _test_functions(self._kind, multipliers, kappa, coefficient)
        return point

    def _coefficient(
        self,
        states: np.ndarray,
        period: float,
        parameters: tuple[float, float],
        multipliers: np.ndarray,
        kappa: float,
    ) -> complex | None:
        if not self._normal_forms:
            return None
        function = self._equation.derivative
        function.parameters[self._parameters[0]].value = parameters[0]
        function.parameters[self._parameters[1]].value = parameters[1]
        normal_form = CycleNormalForm(function, states, period)
        if self._kind == "fold_of_cycles":
            return normal_form.fold()
        if self._kind == "period_doubling":
            return normal_form.flip()
        del multipliers
        if not -1.0 < kappa < 1.0:
            # A neutral saddle, not a torus: no normal form, and no Chenciner test.
            return None
        return normal_form.torus(float(np.arccos(kappa)))


def _split(
    kind: str,
    multipliers: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """The multipliers other than the trivial one and the curve's critical ones."""
    values = list(multipliers)
    trivial = int(np.argmin(np.abs(np.asarray(values) - 1.0)))
    values.pop(trivial)
    remaining = np.asarray(values)
    if kind == "fold_of_cycles":
        drop = [int(np.argmin(np.abs(remaining - 1.0)))]
    elif kind == "period_doubling":
        drop = [int(np.argmin(np.abs(remaining + 1.0)))]
    else:
        residues = np.abs(remaining**2 - 2.0 * kappa * remaining + 1.0)
        drop = list(np.argsort(residues)[:2])
    return np.delete(remaining, drop)


def _test_functions(
    kind: str,
    multipliers: np.ndarray,
    kappa: float,
    coefficient: complex | None,
) -> dict[str, float]:
    """Each codimension-two test function of the curve ``kind`` at one point."""
    others = _split(kind, multipliers, kappa)
    tests: dict[str, float] = {}
    if kind != "fold_of_cycles":
        tests["fold_" + ("flip" if kind == "period_doubling" else "neimark_sacker")] = (
            float(np.real(np.prod(others - 1.0)))
        )
    if kind == "fold_of_cycles":
        tests["fold_flip"] = float(np.real(np.prod(others + 1.0)))
        tests["fold_neimark_sacker"] = _torus_excess(np.concatenate([[1.0], others]))
    elif kind == "period_doubling":
        tests["strong_resonance_1_2"] = float(np.real(np.prod(others + 1.0)))
        tests["flip_neimark_sacker"] = _torus_excess(np.concatenate([[1.0], others]))
    else:
        tests["flip_neimark_sacker"] = float(np.real(np.prod(others + 1.0)))
        tests["double_neimark_sacker"] = _torus_excess(np.concatenate([[1.0], others]))
        for name, value in _RESONANCES:
            tests[name] = kappa - value
    if coefficient is not None:
        tests[_COEFFICIENT_ZEROS[kind]] = float(np.real(coefficient))
    return tests


def _changed(lower: CycleCurvePoint, upper: CycleCurvePoint) -> list[str]:
    """The test functions that change sign between two adjacent points."""
    return [
        name
        for name, before in lower.tests.items()
        if (after := upper.tests.get(name)) is not None and before * after < 0.0
    ]


def _interpolated(
    name: str,
    lower: CycleCurvePoint,
    upper: CycleCurvePoint,
) -> CycleCodim2Point:
    before, after = lower.tests[name], upper.tests[name]
    fraction = before / (before - after)
    return CycleCodim2Point(
        name,
        (
            lower.parameters[0]
            + fraction * (upper.parameters[0] - lower.parameters[0]),
            lower.parameters[1]
            + fraction * (upper.parameters[1] - lower.parameters[1]),
        ),
    )
