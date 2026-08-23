"""Normal-form coefficients for codim-2 and codim-3 classification.

These quantities are built from the multilinear forms of the vector field at a
critical point. The symmetric second- and third-order forms ``B`` and ``C`` are
recovered from directional Taylor coefficients by polarization, so they are exact
whenever the derivative provider is automatic differentiation.

- ``cusp_coefficient`` is the quadratic fold coefficient ``a`` (zero at a cusp).
- ``first_lyapunov`` is the first Lyapunov coefficient ``l1`` at a Hopf point
  (Kuznetsov's invariant formula; zero at a generalized Hopf / Bautin point).
- ``cubic_fold_coefficient`` is the next fold coefficient ``b`` (zero at a
  swallowtail on the cusp curve).
"""

import math

import numpy as np

from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)

_HALF = 0.5
_MINIMUM_DENOMINATOR = 1.0e-9
_SIX = 6.0


class MultilinearForms:
    """Symmetric multilinear derivatives of the field at a fixed point."""

    __slots__ = ["_parameter_a", "_parameter_b", "_problem", "_state"]

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

    def _second(self, direction: np.ndarray) -> np.ndarray:
        # Coefficient 2 equals B(direction, direction) / 2.
        return self._problem.taylor(
            self._state,
            direction,
            2,
            self._parameter_a,
            self._parameter_b,
        )[2]

    def _third(self, direction: np.ndarray) -> np.ndarray:
        # Coefficient 3 equals C(direction, direction, direction) / 6.
        return self._problem.taylor(
            self._state,
            direction,
            3,
            self._parameter_a,
            self._parameter_b,
        )[3]

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


def cusp_coefficient(
    problem: TwoParameterProblem,
    state: np.ndarray,
    parameter_a: float,
    parameter_b: float,
) -> float:
    """Quadratic fold coefficient ``a = (1/2) <p, B(q, q)>``."""
    jacobian = problem.state_jacobian(state, parameter_a, parameter_b)
    right_null = _null_vector(jacobian)
    left_null = _null_vector(jacobian.T)
    left_null = left_null / (left_null @ right_null)
    forms = MultilinearForms(problem, state, parameter_a, parameter_b)
    form = forms.bilinear(right_null, right_null)
    return float((_HALF * (left_null @ form)).real)


def cubic_fold_coefficient(
    problem: TwoParameterProblem,
    state: np.ndarray,
    parameter_a: float,
    parameter_b: float,
) -> float:
    """Cubic fold coefficient ``b = (1/6) <p, C(q, q, q)>`` (zero at swallowtail)."""
    jacobian = problem.state_jacobian(state, parameter_a, parameter_b)
    right_null = _null_vector(jacobian)
    left_null = _null_vector(jacobian.T)
    left_null = left_null / (left_null @ right_null)
    forms = MultilinearForms(problem, state, parameter_a, parameter_b)
    form = forms.trilinear(right_null, right_null, right_null)
    return float(((left_null @ form) / _SIX).real)


def first_lyapunov(
    problem: TwoParameterProblem,
    state: np.ndarray,
    parameter_a: float,
    parameter_b: float,
    omega: float,
) -> float:
    """First Lyapunov coefficient ``l1`` at a Hopf point (Kuznetsov's formula)."""
    jacobian = problem.state_jacobian(state, parameter_a, parameter_b)
    dimension = jacobian.shape[0]
    right = _eigenvector(jacobian, 1j * omega)
    left = _eigenvector(jacobian.T, -1j * omega)
    right = right / np.sqrt(right.conj() @ right)
    left = left / (left.conj() @ right)

    forms = MultilinearForms(problem, state, parameter_a, parameter_b)
    conjugate = right.conj()
    identity = np.eye(dimension)
    h11 = np.linalg.solve(-jacobian, forms.bilinear(right, conjugate))
    h20 = np.linalg.solve(
        2j * omega * identity - jacobian,
        forms.bilinear(right, right),
    )
    g21 = left.conj() @ (
        forms.trilinear(right, right, conjugate)
        + 2.0 * forms.bilinear(right, h11)
        + forms.bilinear(conjugate, h20)
    )
    return float(g21.real / (2.0 * omega))
