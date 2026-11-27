"""Normal-form coefficients for codim-2 and codim-3 classification.

These quantities are built from the multilinear forms of the vector field at a
critical point. The symmetric second- and third-order forms ``B`` and ``C`` are
recovered from directional Taylor coefficients by polarization, so they are exact
whenever the derivative provider is automatic differentiation.

- ``cusp_coefficient`` is the quadratic fold coefficient ``a`` (zero at a cusp),
  with the null vector's sign fixed by an orientation (``fold_null_vectors``).
- ``first_lyapunov`` is the first Lyapunov coefficient ``l1`` at a Hopf point
  (Kuznetsov's invariant formula; zero at a generalized Hopf / Bautin point).
- ``cubic_fold_coefficient`` is the next fold coefficient ``b`` on the center
  manifold (zero at a swallowtail on the cusp curve).
"""

import math

import numpy as np

from discrecontinual_equations.continuation.codim2_normal_form import cusp_normal_form
from discrecontinual_equations.continuation.critical_forms import (
    right_null,
    unit_phase,
)
from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)

_HALF = 0.5
_MINIMUM_DENOMINATOR = 1.0e-9
_SECOND_ORDER = 2
_THIRD_ORDER = 3


class MultilinearForms:
    """Symmetric multilinear derivatives of the field at a fixed point.

    Each directional Taylor coefficient is cached, so polarising many forms that
    share an argument (``B(v, e_k)`` over ``k``, say) evaluates the shared
    direction once.
    """

    __slots__ = ["_cache", "_parameter_a", "_parameter_b", "_problem", "_state"]

    def __init__(
        self,
        problem: TwoParameterProblem,
        state: np.ndarray,
        parameter_a: float,
        parameter_b: float,
    ) -> None:
        self._problem = problem
        self._state = state.astype(complex)
        self._parameter_a = parameter_a
        self._parameter_b = parameter_b
        self._cache: dict[tuple[int, bytes], np.ndarray] = {}

    def _power(self, direction: np.ndarray, order: int) -> np.ndarray:
        # Coefficient m equals D^m f(direction, ..., direction) / m!.
        direction = np.asarray(direction, dtype=complex)
        key = (order, direction.tobytes())
        if key not in self._cache:
            self._cache[key] = self._problem.taylor(
                self._state,
                direction,
                order,
                self._parameter_a,
                self._parameter_b,
            )[order]
        return self._cache[key]

    def _second(self, direction: np.ndarray) -> np.ndarray:
        return self._power(direction, _SECOND_ORDER)

    def _third(self, direction: np.ndarray) -> np.ndarray:
        return self._power(direction, _THIRD_ORDER)

    def form(self, *vectors: np.ndarray) -> np.ndarray:
        """The symmetric ``m``-linear form on ``m`` vectors, by polarisation.

        ``F(v1, ..., vm) = sum over nonempty S of (-1)^(m - |S|) P_m(sum_S v_i)``
        with ``P_m(d) = D^m f(d, ..., d) / m!``.
        """
        order = len(vectors)
        total = np.zeros(self._state.size, dtype=complex)
        for mask in range(1, 2**order):
            chosen = [vectors[i] for i in range(order) if mask >> i & 1]
            sign = -1.0 if (order - len(chosen)) % 2 else 1.0
            total = total + sign * self._power(np.sum(chosen, axis=0), order)
        return total

    def bilinear(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Symmetric second-order form ``B(left, right)``."""
        return self._second(left + right) - self._second(left) - self._second(right)

    def trilinear(
        self,
        first: np.ndarray,
        second: np.ndarray,
        third: np.ndarray,
    ) -> np.ndarray:
        """Symmetric third-order form ``C(first, second, third)``."""
        return (
            self._third(first + second + third)
            - self._third(first + second)
            - self._third(first + third)
            - self._third(second + third)
            + self._third(first)
            + self._third(second)
            + self._third(third)
        )


def _null_vector(matrix: np.ndarray) -> np.ndarray:
    _left, _singular, right = np.linalg.svd(matrix)
    return right[-1].conj()


def _eigenvector(matrix: np.ndarray, eigenvalue: complex) -> np.ndarray:
    values, vectors = np.linalg.eig(matrix)
    index = min(
        range(len(values)),
        key=lambda position: abs(values[position] - eigenvalue),
    )
    return vectors[:, index]


def bogdanov_takens_second_coefficient(
    problem: TwoParameterProblem,
    state: np.ndarray,
    parameter_a: float,
    parameter_b: float,
) -> float:
    """Quadratic cross-coefficient ``b`` of a Bogdanov-Takens normal form.

    In the planar normal form ``x' = y``, ``y' = a x^2 + b x y + ...`` the second
    codimension-three degeneracy is ``b = 0`` (with ``a != 0``); it is the point at
    which the two topological types of the Bogdanov-Takens meet. The coordinate-free
    value uses both Jordan chains,

        ``b = <p1, B(q0, q1)> + <p0, B(q0, q0)>``,

    with ``A q0 = 0``, ``A q1 = q0`` (``<q0, q1> = 0``), ``A^T p1 = 0``
    (``<p1, q1> = 1``) and ``A^T p0 = p1`` (``<p0, q1> = 0``).
    """
    jacobian = problem.state_jacobian(state, parameter_a, parameter_b)
    right_null = _null_vector(jacobian)
    dominant = int(np.argmax(np.abs(right_null)))
    if right_null[dominant].real < 0.0:
        right_null = -right_null
    chain, _res, _rank, _sv = np.linalg.lstsq(jacobian, right_null, rcond=None)
    chain = chain - (right_null @ chain) / (right_null @ right_null) * right_null
    left_null = _null_vector(jacobian.T)
    denominator = left_null @ chain
    if abs(denominator) < _MINIMUM_DENOMINATOR:
        return math.nan
    left_null = left_null / denominator
    left_chain, _r2, _rk2, _s2 = np.linalg.lstsq(jacobian.T, left_null, rcond=None)
    left_chain = left_chain - (left_chain @ chain) * left_null
    forms = MultilinearForms(problem, state, parameter_a, parameter_b)
    cross = forms.bilinear(right_null, chain)
    square = forms.bilinear(right_null, right_null)
    return float(((left_null @ cross) + (left_chain @ square)).real)


def bogdanov_takens_coefficient(
    problem: TwoParameterProblem,
    state: np.ndarray,
    parameter_a: float,
    parameter_b: float,
) -> float:
    """Quadratic normal-form coefficient ``a = (1/2) <p1, B(q0, q0)>`` of a BT.

    At a Bogdanov-Takens point the zero eigenvalue is non-semisimple, so the left
    and right null vectors are orthogonal and the fold normalization ``<p, q0> = 1``
    is singular. The correct normalization uses the Jordan chain: ``A q0 = 0``,
    ``A q1 = q0``, and ``<p1, q1> = 1``. The coefficient vanishes at a degenerate
    Bogdanov-Takens point.
    """
    jacobian = problem.state_jacobian(state, parameter_a, parameter_b)
    right_null = _null_vector(jacobian)
    dominant = int(np.argmax(np.abs(right_null)))
    if right_null[dominant].real < 0.0:
        right_null = -right_null
    chain, _residual, _rank, _singular = np.linalg.lstsq(
        jacobian,
        right_null,
        rcond=None,
    )
    chain = chain - (right_null @ chain) / (right_null @ right_null) * right_null
    left_null = _null_vector(jacobian.T)
    denominator = left_null @ chain
    if abs(denominator) < _MINIMUM_DENOMINATOR:
        return math.nan
    left_null = left_null / denominator
    forms = MultilinearForms(problem, state, parameter_a, parameter_b)
    form = forms.bilinear(right_null, right_null)
    return float((_HALF * (left_null @ form)).real)


def fold_null_vectors(
    jacobian: np.ndarray,
    orientation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Unit right null vector ``q`` and left null vector ``p`` with ``<p, q> = 1``.

    The SVD fixes ``q`` only up to sign, and with ``<p, q> = 1`` the fold
    coefficient ``a = (1/2) <p, B(q, q)>`` is odd in ``q`` (``p`` turns with it), so
    ``<p, q> > 0`` alone leaves the sign of ``a`` to LAPACK. The sign of ``q`` is
    therefore chosen: ``orientation . q > 0`` when an orientation is given (the
    previous point's ``q``, or the fold curve's own null vector, as MatCont's
    bordered ``v_bor^T v = 1`` does), and otherwise the largest component of ``q``
    positive, which is repeatable but not continuous along a curve.
    """
    right = unit_phase(right_null(jacobian).real.astype(complex)).real
    if orientation is not None and float(np.real(orientation) @ right) < 0.0:
        right = -right
    left = right_null(jacobian.T).real
    return right, left / (left @ right)


def cusp_coefficient(
    problem: TwoParameterProblem,
    state: np.ndarray,
    parameter_a: float,
    parameter_b: float,
    orientation: np.ndarray | None = None,
) -> float:
    """Quadratic fold coefficient ``a = (1/2) <p, B(q, q)>`` (Kuznetsov 1999, (2.3)).

    ``|q| = 1`` and ``<p, q> = 1``; the sign of ``q``, on which the sign of ``a``
    depends, follows ``orientation`` as in :func:`fold_null_vectors`.
    """
    jacobian = problem.state_jacobian(state, parameter_a, parameter_b)
    right_null_vector, left_null_vector = fold_null_vectors(jacobian, orientation)
    forms = MultilinearForms(problem, state, parameter_a, parameter_b)
    form = forms.bilinear(right_null_vector, right_null_vector)
    return float((_HALF * (left_null_vector @ form)).real)


def cubic_fold_coefficient(
    problem: TwoParameterProblem,
    state: np.ndarray,
    parameter_a: float,
    parameter_b: float,
) -> float:
    """Cubic fold coefficient ``b`` on the center manifold (zero at a swallowtail).

    ``b = <p, C(q, q, q) + 3 B(q, h2)> / 6``, the cusp normal-form coefficient of
    :func:`.codim2_normal_form.cusp_normal_form` (Kuznetsov 1999, section 5). The
    ``3 B(q, h2)`` share is the center manifold's curvature; it vanishes in one
    dimension but not in general, so ``<p, C(q, q, q)> / 6`` alone puts the
    swallowtail in the wrong place once the critical mode is coupled.
    """
    return cusp_normal_form(problem, state, parameter_a, parameter_b).c


class HopfCoefficients:
    """The pieces of Kuznetsov's invariant formula for ``l1`` at a Hopf point.

    ``A q = i omega q`` with ``|q| = 1``, ``A^T p = -i omega p`` with
    ``<p, q> = 1``, ``h11 = -A^-1 B(q, conj q)``,
    ``h20 = (2 i omega - A)^-1 B(q, q)`` and
    ``g21 = <p, C(q, q, conj q) + 2 B(q, h11) + B(conj q, h20)>``.
    """

    __slots__ = ["forms", "g21", "h11", "h20", "jacobian", "left", "omega", "right"]

    def __init__(
        self,
        problem: TwoParameterProblem,
        state: np.ndarray,
        parameter_a: float,
        parameter_b: float,
        omega: float,
    ) -> None:
        jacobian = problem.state_jacobian(state, parameter_a, parameter_b)
        dimension = jacobian.shape[0]
        right = _eigenvector(jacobian, 1j * omega)
        left = _eigenvector(jacobian.T, -1j * omega)
        right = right / np.sqrt(right.conj() @ right)
        # <p, q> = conj(p) . q = 1: divide by the conjugate of the overlap. The
        # two eig calls fix their phases independently, so it is not real.
        left = left / np.conj(left.conj() @ right)

        forms = MultilinearForms(problem, state, parameter_a, parameter_b)
        conjugate = right.conj()
        identity = np.eye(dimension)
        h11 = np.linalg.solve(-jacobian, forms.bilinear(right, conjugate))
        h20 = np.linalg.solve(
            2j * omega * identity - jacobian,
            forms.bilinear(right, right),
        )
        self.g21 = left.conj() @ (
            forms.trilinear(right, right, conjugate)
            + 2.0 * forms.bilinear(right, h11)
            + forms.bilinear(conjugate, h20)
        )
        self.jacobian = jacobian
        self.right = right
        self.left = left
        self.h11 = h11
        self.h20 = h20
        self.omega = omega
        self.forms = forms

    @property
    def first_lyapunov(self) -> float:
        """``l1 = Re g21 / (2 omega)``."""
        return float(self.g21.real / (2.0 * self.omega))


def first_lyapunov(
    problem: TwoParameterProblem,
    state: np.ndarray,
    parameter_a: float,
    parameter_b: float,
    omega: float,
) -> float:
    """First Lyapunov coefficient ``l1`` at a Hopf point (Kuznetsov's formula).

    ``A q = i omega q``, ``A^T p = -i omega p``, ``<p, q> = <q, q> = 1`` with
    ``<p, q> = conj(p) . q``, and ``l1 = Re G21 / (2 omega)``. MatCont prints
    ``Re G21 / 2``, which is ``omega * l1``. The formula assumes ``omega > 0``;
    ``-omega`` names the same eigenpair (a Hopf curve continued past a
    Bogdanov-Takens point returns it), so the positive frequency is used.
    """
    return HopfCoefficients(
        problem,
        state,
        parameter_a,
        parameter_b,
        abs(omega),
    ).first_lyapunov
