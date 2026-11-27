"""Minimally augmented residual whose zero set is the generalized-Hopf (Bautin) curve.

A generalized Hopf is codimension two, so in three active parameters its locus is a
one-dimensional curve. It is defined by the minimally augmented Hopf system of
Govaerts (*Numerical Methods for Bifurcations of Dynamical Equilibria*, SIAM 2000,
section 4.3.4; MatCont's ``hopf.m``, manual eqs. (61)-(62)) together with the
vanishing of the first Lyapunov coefficient:

    f(u; p_a, p_b, p_c)  = 0
    G_i1j1(u, kappa)     = 0      [[f_u^2 + kappa I, W_bor], [V_bor^T, 0]] (V, G)
    G_i2j2(u, kappa)     = 0                                    = (0, I_2)
    l1(u, kappa)         = 0

with unknowns ``(u, kappa, p_a, p_b)`` and continuation coordinate ``p_c``. On the
curve ``f_u`` has eigenvalues ``+-i omega`` with ``kappa = omega^2``. ``l1`` is
:func:`.normal_form.first_lyapunov` at ``omega = sqrt(kappa)`` (Kuznetsov's
invariant formula, *Elements of Applied Bifurcation Theory*, 3rd ed., (3.20); its
zero set is the planar focal value's).

The state columns of the Jacobian are analytic. ``G_z = -W^T (f_u^2)_z V`` with
``W`` from the transposed bordered system (Govaerts). ``l1`` is differentiated
through the eigenvalue problem: the eigenvector derivative comes from one bordered
solve per column and the adjoint eigenvector's from a single adjoint solve, so a
column costs ``O(n^2)`` after one factorisation, plus a fixed number of
multilinear-form evaluations. Only the three parameter columns are differenced,
centrally, at the relative step ``epsilon max(|p|, 1)`` and over the width actually
stepped (:func:`~discrecontinual_equations.finite_difference.central_column`,
DEQ-159); the Jacobian that picks the pair of ``G`` entries at an accepted point
uses the same rule at ``epsilon = 1e-6``.
Borders are frozen through a Newton solve and adapted at accepted points, where
the pair of ``G`` entries is also re-chosen to keep the system best conditioned.
A degenerate Bautin point is where the second Lyapunov quantity ``l2`` changes
sign along this curve.

Normalisation of ``l1``: ``<p, q> = 1`` is imposed at the *positive* frequency
``+i omega`` (DEQ-96). Before that fix the adjoint was normalised at whichever sign
of ``omega`` the eigen-solver returned, and ``l1`` came out wrong for 31 of 34
random similarity transforms of one system; with ``omega = sqrt(kappa)`` here the
frequency is positive by construction. ``l1`` is ``Re G21 / (2 omega)``;
MatCont prints ``Re G21 / 2`` (manual eq. (40)), ``omega`` times this. Only the
zero set matters for the curve, and it is the same.

CONTENT's minimally extended Bautin system (Govaerts, Kuznetsov and Sijnave, eqs.
(7.2)-(7.4)) asks for ``det G = 0``, one equation; here two entries of ``G`` are set
to zero, as MatCont's ``hopf.m`` does (manual eq. (61)). At a Hopf point
``f_u^2 + kappa I`` has a two-dimensional kernel and the whole of ``G`` vanishes, so
both conditions hold; the pair of entries is the better conditioned choice.

Limits: ``kappa > 0`` is assumed (``omega = sqrt(kappa)``); a curve that runs into a
neutral saddle (``kappa < 0``) or a Bogdanov-Takens point (``kappa = 0``) leaves the
domain of the ``l1`` row. The Jacobian columns of ``l1`` are exact only up to the
finite-difference columns of the three parameters.

References:
- Govaerts, W., Kuznetsov, Yu. A., Meijer, H. G. E. et al. "MATCONT: Continuation
  toolbox for ODEs in Matlab", manual for MatCont 7.1 (August 2019), section 8.2.1, eqs.
  (61)-(62): the minimally extended Hopf system and its bordered ``G``; eq. (40):
  ``l1``.
- Govaerts, W., Kuznetsov, Yu. A. and Sijnave, B. "Continuation of codimension-2
  equilibrium bifurcations in CONTENT", in *Numerical Methods for Bifurcation
  Problems and Large-Scale Dynamical Systems*, IMA Vol. Math. Appl., Springer
  (2000) 163-184, section 7.1, eqs. (7.1)-(7.4): the
  minimally extended generalized-Hopf (Bautin) curve with ``l1 = 0``.
- Kuznetsov, Yu. A. "Numerical normalization techniques for all codim 2 bifurcations of
  equilibria in ODE's", SIAM J. Numer. Anal. 36(4) (1999) 1104-1124,
  doi:10.1137/S0036142998335005: eqs. (7.2), (7.3), (7.6), the vectors ``h20``, ``h11``
  and ``G21`` of ``l1``; eq. (2.5), the invariant formula for ``l1``.
"""

import itertools

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from discrecontinual_equations.continuation.bordered import BorderedMatrix
from discrecontinual_equations.continuation.cusp_curve import Codim3Decoded
from discrecontinual_equations.continuation.normal_form import HopfCoefficients
from discrecontinual_equations.continuation.residual import DifferentiableResidual
from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)
from discrecontinual_equations.finite_difference import central_column
from discrecontinual_equations.parameter import Parameter

_BORDER = 2
_UNKNOWNS_AFTER_STATE = 3
_SELECTION_EPSILON = 1.0e-6
_ENTRIES = [(0, 0), (0, 1), (1, 0), (1, 1)]


class _HopfLinearization:
    """Everything the residual and its state columns share at one point."""

    __slots__ = [
        "bordered",
        "coefficients",
        "jacobian",
        "left",
        "omega",
        "residual",
        "right",
        "schur",
    ]

    def __init__(
        self,
        problem: TwoParameterProblem,
        state: np.ndarray,
        borders: tuple[np.ndarray, np.ndarray],
    ) -> None:
        n = state.size - _UNKNOWNS_AFTER_STATE
        equilibrium = state[:n]
        kappa = float(state[n])
        parameter_a, parameter_b = float(state[n + 1]), float(state[n + 2])
        right_border, left_border = borders
        self.jacobian = problem.state_jacobian(equilibrium, parameter_a, parameter_b)
        square = self.jacobian @ self.jacobian + kappa * np.eye(n)
        self.bordered = BorderedMatrix(square, left_border, right_border)
        unit = np.zeros((n + _BORDER, _BORDER))
        unit[n:, :] = np.eye(_BORDER)
        solution = self.bordered.solve(unit)
        self.right = solution[:n]
        self.schur = solution[n:]
        self.left = self.bordered.solve_transposed(unit)[:n]
        self.omega = float(np.sqrt(kappa))
        self.coefficients = HopfCoefficients(
            problem,
            equilibrium,
            parameter_a,
            parameter_b,
            self.omega,
        )
        self.residual = problem.evaluate(equilibrium, parameter_a, parameter_b)


class GeneralizedHopfCurveResidual(DifferentiableResidual):
    """Minimally augmented Hopf system with the condition ``l1 = 0``.

    Args:
        problem: The two-parameter problem (``p_a``, ``p_b``).
        dimension: The state dimension ``n``.
        right_border: ``n x 2`` border ``V_bor``, orthonormalised.
        left_border: ``n x 2`` border ``W_bor``, orthonormalised.
        third: The third active parameter ``p_c``, the curve coordinate; it is
            set on every evaluation.

    References:
    - MatCont manual (2019), eqs. (61)-(62).
    - Govaerts, Kuznetsov & Sijnave (2000), section 7.1.
    """

    __slots__ = [
        "_cache",
        "_dimension",
        "_entries",
        "_left_border",
        "_problem",
        "_right_border",
        "_third",
    ]

    def __init__(
        self,
        problem: TwoParameterProblem,
        dimension: int,
        right_border: np.ndarray,
        left_border: np.ndarray,
        third: Parameter,
    ) -> None:
        self._problem = problem
        self._dimension = dimension
        self._right_border = np.linalg.qr(right_border)[0]
        self._left_border = np.linalg.qr(left_border)[0]
        self._third = third
        self._entries = [_ENTRIES[0], _ENTRIES[3]]
        self._cache: tuple[bytes, _HopfLinearization] | None = None

    def _linearize(self, state: np.ndarray, parameter: float) -> _HopfLinearization:
        self._third.value = float(parameter)
        key = np.append(state, parameter).tobytes()
        if self._cache is None or self._cache[0] != key:
            borders = (self._right_border, self._left_border)
            self._cache = (key, _HopfLinearization(self._problem, state, borders))
        return self._cache[1]

    def evaluate(self, state: np.ndarray, parameter: float) -> np.ndarray:
        """Return ``(f, G_i1j1, G_i2j2, l1)``.

        Args:
            state: ``(u, kappa, p_a, p_b)``.
            parameter: The continuation coordinate ``p_c``.

        Returns:
            The ``n + 3`` residual rows.
        """
        point = self._linearize(np.asarray(state, dtype=float), parameter)
        return self._rows(point, self._entries)

    @staticmethod
    def _rows(
        point: _HopfLinearization,
        entries: list[tuple[int, int]],
    ) -> np.ndarray:
        schur = [point.schur[i, j] for i, j in entries]
        lyapunov = point.coefficients.first_lyapunov
        return np.concatenate([point.residual, schur, [lyapunov]])

    def extended_jacobian(
        self,
        state: np.ndarray,
        parameter: float,
        epsilon: float,
    ) -> np.ndarray:
        """Return the ``(n + 3) x (n + 4)`` Jacobian in ``(u, kappa, p_a, p_b, p_c)``.

        Args:
            state: ``(u, kappa, p_a, p_b)``.
            parameter: The continuation coordinate ``p_c``.
            epsilon: The relative central-difference step for the three
                parameter columns: ``p`` steps ``epsilon max(|p|, 1)``. The
                state and ``kappa`` columns are analytic.

        Returns:
            The extended Jacobian.
        """
        return self._jacobian(
            np.asarray(state, dtype=float),
            parameter,
            epsilon,
            self._entries,
        )

    def _jacobian(
        self,
        state: np.ndarray,
        parameter: float,
        epsilon: float,
        entries: list[tuple[int, int]],
    ) -> np.ndarray:
        n = self._dimension
        point = self._linearize(state, parameter)
        rows = n + len(entries) + 1
        jacobian = np.zeros((rows, state.size + 1))
        jacobian[:n, :n] = point.jacobian
        schur_state = self._schur_state_columns(point)
        schur_kappa = -point.left.T @ point.right
        for row, (i, j) in enumerate(entries, start=n):
            jacobian[row, :n] = schur_state[i, j]
            jacobian[row, n] = schur_kappa[i, j]
        jacobian[-1, : n + 1] = _lyapunov_gradient(point)
        for column in range(n + 1, state.size + 1):
            jacobian[:, column] = self._difference(
                state,
                parameter,
                column,
                epsilon,
                entries,
            )
        return jacobian

    def _schur_state_columns(self, point: _HopfLinearization) -> np.ndarray:
        """``d G_ij / d u_k = -(W_i . B(A V_j, e_k) + (A^T W_i) . B(V_j, e_k))``."""
        n = self._dimension
        forms = point.coefficients.forms
        right, left = point.right, point.left
        image = point.jacobian @ right
        left_image = point.jacobian.T @ left
        units = np.eye(n)
        columns = np.zeros((_BORDER, _BORDER, n))
        for k in range(n):
            for j in range(_BORDER):
                through_image = forms.bilinear(image[:, j], units[k]).real
                through_right = forms.bilinear(right[:, j], units[k]).real
                columns[:, j, k] = -(
                    left.T @ through_image + left_image.T @ through_right
                )
        return columns

    def _difference(
        self,
        state: np.ndarray,
        parameter: float,
        column: int,
        epsilon: float,
        entries: list[tuple[int, int]],
    ) -> np.ndarray:
        def rows(point: np.ndarray, coordinate: float) -> np.ndarray:
            return self._rows(self._linearize(point, coordinate), entries)

        difference = central_column(rows, state, parameter, column, epsilon)
        self._third.value = float(parameter)
        return difference

    def adapt(self, state: np.ndarray, parameter: float) -> None:
        """Move the borders to the null spaces, then pick the best pair of ``G``.

        The borders become orthonormal bases of the computed ``V`` and ``W``. Of
        the six pairs of entries of ``G``, the one whose augmented Jacobian has the
        largest smallest singular value is kept until the next accepted point.

        Args:
            state: The accepted point ``(u, kappa, p_a, p_b)``.
            parameter: Its ``p_c``.
        """
        state = np.asarray(state, dtype=float)
        point = self._linearize(state, parameter)
        self._right_border = np.linalg.qr(point.right)[0]
        self._left_border = np.linalg.qr(point.left)[0]
        self._cache = None
        full = self._jacobian(state, parameter, _SELECTION_EPSILON, _ENTRIES)
        n = self._dimension
        best = -1.0
        for first, second in itertools.combinations(range(len(_ENTRIES)), 2):
            rows = [*range(n), n + first, n + second, n + len(_ENTRIES)]
            smallest = float(np.linalg.svd(full[rows], compute_uv=False)[-1])
            if smallest > best:
                best = smallest
                self._entries = [_ENTRIES[first], _ENTRIES[second]]
        self._cache = None

    def decode(self, state: np.ndarray, parameter: float) -> Codim3Decoded:
        """Recover the original-system point from the augmented vector.

        Args:
            state: ``(u, kappa, p_a, p_b)``.
            parameter: ``p_c``.

        Returns:
            The equilibrium and the three parameters.
        """
        n = self._dimension
        self._third.value = float(parameter)
        return Codim3Decoded(
            state=state[:n],
            parameter_a=float(state[n + 1]),
            parameter_b=float(state[n + 2]),
            parameter_c=float(parameter),
        )


def _lyapunov_gradient(point: _HopfLinearization) -> np.ndarray:
    """``d l1 / d (u, kappa)`` through the eigenvalue problem, column by column.

    With ``l1 = Re g21 / (2 omega)``, ``omega = sqrt(kappa)``, ``A q = lambda q``,
    ``r^T A = lambda r^T``, ``r^T q = 1``, ``|q| = 1`` and ``A_k = B(., e_k)``:

    - ``q_k = x - Re(q^H x) q`` with ``K (x, .) = (-A_k q, 0)`` and
      ``K = [[A - lambda, q], [r^T, 0]]``. The eigenvalue's own derivative
      ``lambda_k = r^T A_k q`` would add ``lambda_k q`` to the right-hand side;
      the border variable absorbs exactly that, so it is left out. The phase of
      ``q_k`` is free: ``l1`` does not depend on it;
    - ``h11_k = -A^-1 (C(q, conj q, e_k) + 2 Re B(q_k, conj q) + A_k h11)``,
      ``h20_k = (2 i omega - A)^-1 (C(q, q, e_k) + 2 B(q, q_k) + A_k h20)``;
    - ``r_k^T R = -r^T A_k y - g21 r^T q_k`` with ``K (y, .) = (R, 0)``, so the
      adjoint eigenvector is never differentiated;
    - ``g21_k = r_k^T R + r^T R_k`` with ``R_k`` the derivative of
      ``C(q, q, conj q) + 2 B(q, h11) + B(conj q, h20)``.

    ``kappa`` enters only through ``omega`` in ``h20`` and the ``1 / (2 omega)``.
    """
    hopf = point.coefficients
    forms = hopf.forms
    matrix = hopf.jacobian
    n = matrix.shape[0]
    omega = hopf.omega
    q, r = hopf.right, hopf.left.conj()
    qbar = q.conj()
    h11, h20, g21 = hopf.h11, hopf.h20, hopf.g21
    eigenvalue = complex(r @ matrix @ q)
    resonant = 2j * omega * np.eye(n) - matrix
    eigen_border = BorderedMatrix(matrix - eigenvalue * np.eye(n), q, r)
    plain = lu_factor(matrix)
    doubled = lu_factor(resonant)
    square = forms.form(q, q, qbar) + 2.0 * forms.form(q, h11) + forms.form(qbar, h20)
    adjoint = eigen_border.solve(np.append(square, 0.0))[:n]
    units = np.eye(n)
    gradient = np.zeros(n + 1)
    for k in range(n):
        unit = units[k].astype(complex)
        shifted = eigen_border.solve(np.append(-forms.form(q, unit), 0.0))[:n]
        q_k = shifted - np.real(np.vdot(q, shifted)) * q
        qbar_k = q_k.conj()
        h11_k = -lu_solve(
            plain,
            forms.form(q, qbar, unit)
            + 2.0 * forms.form(q_k, qbar).real
            + forms.form(h11, unit),
        )
        h20_k = lu_solve(
            doubled,
            forms.form(q, q, unit) + 2.0 * forms.form(q, q_k) + forms.form(h20, unit),
        )
        square_k = (
            forms.form(q, q, qbar, unit)
            + 2.0 * forms.form(q, qbar, q_k)
            + forms.form(q, q, qbar_k)
            + 2.0 * forms.form(q, h11, unit)
            + 2.0 * forms.form(q_k, h11)
            + 2.0 * forms.form(q, h11_k)
            + forms.form(qbar, h20, unit)
            + forms.form(qbar_k, h20)
            + forms.form(qbar, h20_k)
        )
        adjoint_term = -(r @ forms.form(adjoint, unit)) - g21 * (r @ q_k)
        gradient[k] = (adjoint_term + r @ square_k).real / (2.0 * omega)
    rate_omega = 1.0 / (2.0 * omega)
    h20_kappa = lu_solve(doubled, -2j * rate_omega * h20)
    g21_kappa = r @ forms.form(qbar, h20_kappa)
    gradient[n] = g21_kappa.real / (2.0 * omega) - g21.real * rate_omega / (
        2.0 * omega**2
    )
    return gradient
