"""Normal-form coefficients on the center manifold at codim-2 equilibria.

The codim-2 test functions (:mod:`.codim2`) say *where* a Bogdanov-Takens,
zero-Hopf or Hopf-Hopf point is; the coefficients computed here say *which* one it
is. They come from Kuznetsov's projection method (Yu. A. Kuznetsov, "Numerical
normalization techniques for all codim 2 bifurcations of equilibria in ODE's",
SIAM J. Numer. Anal. 36 (1999) 1104-1124): the center manifold ``x = H(w)`` and
the restricted normal form ``w' = G(w)`` are solved together from the homological
equation ``H_w(w) G(w) = F(H(w))`` order by order. At each order the coefficient of
``H`` solves a linear system ``L h = R``; where ``L`` is singular the Fredholm
condition ``<p, R> = 0`` gives the normal-form coefficient, and the bordered system
gives ``h``. Everything is built from the critical eigenvectors and the exact
multilinear forms of :class:`.critical_forms.CriticalForms`, so the formulas hold
in any dimension and the center manifold's curvature (through the ``h`` terms) is
included.

Eigenvectors are normalised as MatCont does - unit length, and ``q1 . q0 = 0`` at a
Bogdanov-Takens point - with the phase fixed by :func:`.critical_forms.unit_phase`.
The raw coefficients depend on that normalisation; the classifying quantities
(``s``, ``theta``, ``simple``, ...) do not.

The Bogdanov-Takens unfolding follows Bosschaert and Kuznetsov, "Interplay between
normal forms and center manifold reduction for homoclinic predictors near
Bogdanov-Takens bifurcation", SIAM J. Appl. Dyn. Syst. (2024), arXiv:2109.12570,
section 2.2.3 (through the ``beta2^2`` term of the parameter map).

Errata and decisions carried here:

- Lorenz-84 ``G300``: Kuznetsov (1999), section 10, prints
  ``G300 = 25602814720/36130026241 = 0.7086`` for the "new" Lorenz model. The
  paper's own section 8 formula ``G300 = <p0, C(q0, q0, q0) + 3 B(q0, h200)>``,
  which matches the ``G300/6`` of its normal form (3.4), gives exactly three times
  that, ``2.1259``, on the paper's printed ``q0``, ``p0`` (the model has no cubic
  terms, so the printed value is ``<p0, B(q0, h200)>``: the factor 3 is dropped).
  The same computation reproduces every other printed coefficient exactly, and
  this module and an independent Poincare-Dulac reference agree. The printed
  ``e(0) = 0.2106514716`` is the formula at the printed ``G300``; at the
  consistent value it is ``1.6503``. The sign of ``e(0)``, and the paper's
  conclusion (``s = sign(b c) = 1``, ``theta < 0``), are unchanged.
- The cubic cusp coefficient includes the center manifold's ``3 B(q, h2)``
  (Kuznetsov 1999, section 5); dropping it is right only in one dimension (DEQ-83).
- Orientation: ``c`` here is independent of the sign of ``q``, because ``p`` is
  rescaled so that ``<p, q> = 1`` and ``c`` is even in ``q``. The *quadratic* fold
  coefficient ``a = <p, B(q, q)>/2`` is odd in ``q``, so its sign along a fold
  curve is fixed by continuity of ``q``, as MatCont does, not by requiring
  ``<p, q> > 0`` (DEQ-102; that lives in :mod:`.normal_form` and the detectors).

Limits: the Bogdanov-Takens unfolding stops at ``beta2^2``; the double Hopf is
cubic only (the fifth-order coefficients of the difficult case are not computed);
the zero-Hopf ``e`` is exact but the higher-order terms that can change the
topology of the unfolding (Kuznetsov 1999, section 3 (iv)) are not computed.

References:
- Kuznetsov, Yu. A. "Numerical normalization techniques for all codim 2 bifurcations of
  equilibria in ODE's", SIAM J. Numer. Anal. 36(4) (1999) 1104-1124,
  doi:10.1137/S0036142998335005: section 3, the normal forms (3.1)-(3.6) and the
  resonance condition (3.5); section 4, eqs. (4.4)-(4.6); sections 5-9, the coefficient
  formulas; section 10, the Lorenz-84 example.
- Bosschaert, M. M. and Kuznetsov, Yu. A. "Interplay between normal forms and
  center manifold reduction for homoclinic predictors near Bogdanov-Takens
  bifurcation", arXiv:2109.12570v1 (2021): eq. (8), the unfolding; sections 2.2.3
  and 2.2.7, the parameter map ``K10``, ``K01``, ``K02``.
- Govaerts, W., Kuznetsov, Yu. A., Meijer, H. G. E. et al. "MATCONT: Continuation
  toolbox for ODEs in Matlab", manual for MatCont 7.1 (August 2019), section 6.3, eq.
  (40): the ``l1 = Re G21 / 2`` MatCont prints.
"""

import math
from dataclasses import dataclass

import numpy as np

from discrecontinual_equations.continuation.critical_forms import (
    CriticalForms,
    adjoint,
    bordered_solve,
    eigenvector,
    inner,
    right_null,
    unit_phase,
)
from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)

_IMAGINARY = 1.0e-8
_HALF = 0.5
_HOMOCLINIC = 49.0 / 25.0
_NEWTON_STEPS = 50
_NEWTON_TOLERANCE = 1.0e-15
_QUARTER_TURN = np.array([[0.0, -1.0], [1.0, 0.0]])
_RESONANCE_ORDER = 5
_RESONANCE_TOLERANCE = 1.0e-6


def _real_unit(vector: np.ndarray) -> np.ndarray:
    return unit_phase(vector.real.astype(complex)).real


def _critical_pairs(jacobian: np.ndarray) -> list[complex]:
    """Eigenvalues with positive imaginary part, nearest the imaginary axis first."""
    values = np.linalg.eigvals(jacobian)
    upper = [value for value in values if value.imag > _IMAGINARY]
    return sorted(upper, key=lambda value: abs(value.real))


@dataclass(frozen=True)
class HopfNormalForm:
    """``w' = i omega w + G21/2 w |w|^2`` (Kuznetsov 1999, eq. 7.6)."""

    omega: float
    q: np.ndarray
    p: np.ndarray
    g21: complex

    @property
    def first_lyapunov(self) -> float:
        """``Re G21 / (2 omega)``; negative means supercritical.

        This is the convention of :func:`.normal_form.first_lyapunov` and of the
        normal form time-scaled to unit frequency. ``p`` is normalised at the
        positive frequency ``+i omega`` (DEQ-96).
        """
        return self.g21.real / (2.0 * self.omega)

    @property
    def matcont_lyapunov(self) -> float:
        """``Re G21 / 2``, as MatCont prints it (Kuznetsov 1999, eq. 2.5)."""
        return _HALF * self.g21.real


def hopf_normal_form(
    problem: TwoParameterProblem,
    state: np.ndarray,
    parameter_a: float,
    parameter_b: float,
) -> HopfNormalForm:
    """Kuznetsov (1999), eqs. (7.2), (7.3), (7.6).

    Args:
        problem: The two-parameter problem.
        state: The Hopf equilibrium.
        parameter_a: The first active parameter there.
        parameter_b: The second active parameter there.

    Returns:
        ``omega``, the eigenvectors and ``G21`` at the pair nearest the axis.

    Raises:
        IndexError: If the Jacobian has no eigenvalue with positive imaginary part.

    References:
    - Kuznetsov (1999), eqs. (7.2), (7.3), (7.6); eq. (2.5) for ``l1``.
    """
    jacobian = problem.state_jacobian(state, parameter_a, parameter_b)
    size = jacobian.shape[0]
    omega = float(_critical_pairs(jacobian)[0].imag)
    q = unit_phase(eigenvector(jacobian, 1j * omega))
    p = adjoint(jacobian, 1j * omega, q)
    forms = CriticalForms(problem, state, parameter_a, parameter_b)
    h20 = np.linalg.solve(2j * omega * np.eye(size) - jacobian, forms.state(q, q))
    h11 = np.linalg.solve(-jacobian, forms.state(q, q.conj()))
    g21 = inner(
        p,
        forms.state(q, q, q.conj())
        + forms.state(q.conj(), h20)
        + 2.0 * forms.state(q, h11),
    )
    return HopfNormalForm(omega, q, p, g21)


@dataclass(frozen=True)
class BogdanovTakensUnfolding:
    """The parameter map ``alpha = K(beta)`` onto the unfolding of the BT.

    In the normal form ``w0' = w1``, ``w1' = beta1 + beta2 w1 + a w0^2 + b w0 w1``
    (Bosschaert-Kuznetsov eq. 8) the fold is ``beta1 = 0``, the Hopf is
    ``beta1 = -(a/b^2) beta2^2`` and the homoclinic is
    ``beta1 = -(49/25)(a/b^2) beta2^2 + O(beta2^3)``; the Hopf and homoclinic lie on
    the side ``sign(beta2) = sign(ab)``. The map is kept through ``beta2^2``:

        alpha = alpha0 + K10 beta1 + K01 beta2 + K02 beta2^2 / 2.

    Along the curves ``beta1 = O(beta2^2)``, so that is every term of order two. The
    ``K01`` components of ``K10`` and ``K02`` (``delta2`` and ``delta3`` in the
    paper) are left out: they only re-parametrise each curve along ``K01`` at second
    order, so the curves as point sets are still right to ``O(beta2^3)``.
    """

    origin: np.ndarray
    k10: np.ndarray
    k01: np.ndarray
    k02: np.ndarray
    a: float
    b: float

    @property
    def side(self) -> float:
        """The sign of ``beta2`` on the Hopf and homoclinic branches."""
        return math.copysign(1.0, self.a * self.b)

    def parameters(self, beta1: float, beta2: float) -> np.ndarray:
        """The original parameters ``(p_a, p_b)`` at normal-form ``(beta1, beta2)``."""
        return (
            self.origin
            + self.k10 * beta1
            + self.k01 * beta2
            + _HALF * self.k02 * beta2 * beta2
        )

    def fold(self, beta2: float) -> np.ndarray:
        """A point of the predicted fold curve."""
        return self.parameters(0.0, beta2)

    def hopf(self, beta2: float) -> np.ndarray:
        """A point of the predicted Hopf curve (take ``beta2`` on :attr:`side`)."""
        return self.parameters(-self.a / self.b**2 * beta2 * beta2, beta2)

    def homoclinic(self, beta2: float) -> np.ndarray:
        """A point of the predicted homoclinic curve (``beta2`` on :attr:`side`)."""
        return self.parameters(-_HOMOCLINIC * self.a / self.b**2 * beta2 * beta2, beta2)

    def normal_form_parameters(self, alpha: np.ndarray) -> tuple[float, float]:
        """Invert the map: the ``(beta1, beta2)`` of original parameters ``alpha``."""
        linear = np.column_stack([self.k10, self.k01])
        beta = np.linalg.solve(linear, np.asarray(alpha, dtype=float) - self.origin)
        for _ in range(_NEWTON_STEPS):
            defect = self.parameters(beta[0], beta[1]) - alpha
            if np.linalg.norm(defect) < _NEWTON_TOLERANCE:
                break
            jacobian = np.column_stack([self.k10, self.k01 + self.k02 * beta[1]])
            beta = beta - np.linalg.solve(jacobian, defect)
        return float(beta[0]), float(beta[1])

    def kuznetsov_parameters(self, alpha: np.ndarray) -> tuple[float, float]:
        """The parameters of Kuznetsov's form ``w1' = b1 + b2 w0 + w0^2 + s w0 w1``.

        Scaling ``w0 -> (a/b^2) w0`` and time by ``|b/a|`` takes the normal form to
        ``a = 1, b = s`` with ``beta1 -> b^4 beta1 / a^3``, ``beta2 -> |b/a| beta2``;
        shifting ``w0`` then trades the ``beta2 w1`` term for ``b2 w0``. In these
        coordinates the fold is ``b1 = b2^2/4``, the Hopf ``b1 = 0`` and the
        homoclinic ``b1 = -(6/25) b2^2``.
        """
        beta1, beta2 = self.normal_form_parameters(alpha)
        scaled1 = self.b**4 * beta1 / self.a**3
        scaled2 = abs(self.b / self.a) * beta2
        return scaled1 + scaled2 * scaled2, -2.0 * self.side * scaled2


@dataclass(frozen=True)
class BogdanovTakensNormalForm:
    """``w0' = w1``, ``w1' = a w0^2 + b w0 w1`` (Kuznetsov 1999, eq. 3.2)."""

    a: float
    b: float
    q0: np.ndarray
    q1: np.ndarray
    p0: np.ndarray
    p1: np.ndarray
    unfolding: BogdanovTakensUnfolding

    @property
    def s(self) -> int:
        """``sign(ab)``: which of the two topological types of the unfolding."""
        return int(np.sign(self.a * self.b))


def bogdanov_takens_normal_form(
    problem: TwoParameterProblem,
    state: np.ndarray,
    parameter_a: float,
    parameter_b: float,
) -> BogdanovTakensNormalForm:
    """Kuznetsov (1999), section 6, and the unfolding of Bosschaert-Kuznetsov.

    ``A q0 = 0``, ``A q1 = q0``, ``A^T p1 = 0``, ``A^T p0 = p1``, with
    ``<p_i, q_j> = delta_ij`` and MatCont's ``|q0| = 1``, ``q1 . q0 = 0``. Then
    ``a = <p1, B(q0, q0)>/2`` and ``b = <p0, B(q0, q0)> + <p1, B(q0, q1)>``.

    Args:
        problem: The two-parameter problem.
        state: The Bogdanov-Takens equilibrium.
        parameter_a: The first active parameter there.
        parameter_b: The second active parameter there.

    Returns:
        ``a``, ``b``, the Jordan chains and the second-order unfolding.

    References:
    - Kuznetsov (1999), section 6, eqs. (6.1)-(6.3), for ``a`` and ``b``.
    - Bosschaert & Kuznetsov (2021), sections 2.2.3 and 2.2.7, for the unfolding.
    """
    jacobian = problem.state_jacobian(state, parameter_a, parameter_b)
    q0 = _real_unit(right_null(jacobian))
    p1 = right_null(jacobian.T).real
    q1 = bordered_solve(jacobian, p1, q0, q0).real
    p1 = p1 / (p1 @ q1)
    p0 = bordered_solve(jacobian.T, q0, q1, p1).real
    forms = CriticalForms(problem, state, parameter_a, parameter_b)
    square = forms.state(q0, q0).real
    a = _HALF * float(p1 @ square)
    b = float(p0 @ square + p1 @ forms.state(q0, q1).real)
    unfolding = _unfolding(
        forms,
        jacobian,
        (q0, q1, p0, p1),
        (a, b),
        np.array([parameter_a, parameter_b], dtype=float),
    )
    return BogdanovTakensNormalForm(a, b, q0, q1, p0, p1, unfolding)


def _unfolding(
    forms: CriticalForms,
    jacobian: np.ndarray,
    vectors: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    coefficients: tuple[float, float],
    origin: np.ndarray,
) -> BogdanovTakensUnfolding:
    """Bosschaert-Kuznetsov, section 2.2.3 and the ``K02`` of section 2.2.7.

    The linear terms of the homological equation in ``beta`` give
    ``-A H0001 = J1 K01`` and ``-A H0010 = J1 K10 - q1``; the ``w beta2`` terms fix
    the scale ``delta1`` of ``K01`` and the ``q0`` component ``gamma3`` of ``H0001``;
    the ``beta2^2`` term gives ``K02`` through the solvability condition with
    ``p1 J1 K10 = 1``.
    """
    q0, q1, p0, p1 = vectors
    a, b = coefficients
    gradient = forms.parameter_jacobian()
    nu = p1 @ gradient
    k10 = nu / (nu @ nu)
    k01_hat = _QUARTER_TURN @ k10
    h0001_hat = -_solve(jacobian, p1, q0, gradient @ k01_hat)
    along0 = forms.state(h0001_hat, q0).real + forms.mixed([q0], [k01_hat]).real
    along1 = forms.state(h0001_hat, q1).real + forms.mixed([q1], [k01_hat]).real
    gamma3 = -(p1 @ along0) / (2.0 * a)
    delta1 = 1.0 / (p1 @ along1 + p0 @ along0 + gamma3 * b)
    k01 = delta1 * k01_hat
    h0001 = delta1 * (h0001_hat + gamma3 * q0)
    quadratic = (
        2.0 * forms.mixed([h0001], [k01]).real
        + forms.state(h0001, h0001).real
        + forms.mixed([], [k01, k01]).real
    )
    k02 = -(p1 @ quadratic) * k10
    return BogdanovTakensUnfolding(origin, k10, k01, k02, a, b)


def _solve(
    jacobian: np.ndarray,
    p1: np.ndarray,
    q0: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """``A^INV`` of Bosschaert-Kuznetsov: bordered by ``p1`` and ``q0``."""
    return bordered_solve(jacobian, p1, q0, right).real


@dataclass(frozen=True)
class ZeroHopfNormalForm:
    """The fold-Hopf normal form on the center manifold (Kuznetsov 1999, eq. 3.4).

    ``w0' = G200/2 w0^2 + G011 |w1|^2 + G300/6 w0^3 + G111 w0 |w1|^2``,
    ``w1' = i omega w1 + G110 w0 w1 + G210/2 w0^2 w1 + G021/2 w1 |w1|^2``.
    """

    omega: float
    q0: np.ndarray
    q1: np.ndarray
    p0: np.ndarray
    p1: np.ndarray
    g200: float
    g110: complex
    g011: float
    g300: float
    g111: float
    g210: complex
    g021: complex

    @property
    def b(self) -> float:
        """``b(0) = G200 / 2`` of ``u' = b u^2 + c |z|^2``."""
        return _HALF * self.g200

    @property
    def c(self) -> float:
        """``c(0) = G011``."""
        return self.g011

    @property
    def d(self) -> complex:
        """``d(0) = G110 - i omega G300 / (3 G200)`` of ``z' = ... + d u z``."""
        return self.g110 - 1j * self.omega * self.g300 / (3.0 * self.g200)

    @property
    def e(self) -> float:
        """``e(0)`` of ``z' = ... + e u^2 z`` (Kuznetsov 1999, section 3 (iv)).

        For the Lorenz-84 example the paper prints ``0.2106514716``, which is
        this formula at its misprinted ``G300``; the consistent value is
        ``1.6503`` (see the module docstring).
        """
        bracket = (
            self.g210
            + self.g110
            * (
                self.g021.real / self.g011
                - self.g300 / self.g200
                + self.g111 / self.g011
            )
            - self.g021 * self.g200 / (2.0 * self.g011)
        )
        return _HALF * bracket.real

    @property
    def s(self) -> int:
        """``sign(b c)``, the ``s`` of ``xi' = beta1 + xi^2 + s |zeta|^2``."""
        return int(np.sign(self.b * self.c))

    @property
    def theta(self) -> float:
        """``Re d / b``: the coefficient of ``xi zeta`` after scaling ``xi = b u``."""
        return self.d.real / self.b

    @property
    def torus(self) -> bool:
        """Whether the truncated unfolding has a Neimark-Sacker curve (``s theta < 0``).

        In amplitudes ``xi' = beta1 + xi^2 + s r^2``, ``r' = r (beta2 + theta xi)``
        the cycle ``(xi, r) = (-beta2/theta, r*)`` has Jacobian
        ``[[2 xi, 2 s r], [theta r, 0]]``; its trace vanishes at ``xi = 0`` with
        determinant ``-2 s theta r^2``, positive exactly when ``s theta < 0``. That
        Hopf of the amplitude system is the torus bifurcation of the full system.
        """
        return self.s * self.theta < 0.0


def zero_hopf_normal_form(
    problem: TwoParameterProblem,
    state: np.ndarray,
    parameter_a: float,
    parameter_b: float,
) -> ZeroHopfNormalForm:
    """Kuznetsov (1999), section 8: eqs. (8.2)-(8.6) and the cubic coefficients.

    Args:
        problem: The two-parameter problem.
        state: The zero-Hopf equilibrium.
        parameter_a: The first active parameter there.
        parameter_b: The second active parameter there.

    Returns:
        The quadratic and cubic coefficients ``G200 ... G021`` of (3.4).

    Raises:
        IndexError: If the Jacobian has no eigenvalue with positive imaginary part.

    References:
    - Kuznetsov (1999), eq. (3.4) and section 3 (iv), eqs. (8.2)-(8.6).
    """
    jacobian = problem.state_jacobian(state, parameter_a, parameter_b)
    size = jacobian.shape[0]
    identity = np.eye(size)
    omega = float(_critical_pairs(jacobian)[0].imag)
    q0 = _real_unit(right_null(jacobian))
    p0 = right_null(jacobian.T).real
    p0 = p0 / (p0 @ q0)
    q1 = unit_phase(eigenvector(jacobian, 1j * omega))
    p1 = adjoint(jacobian, 1j * omega, q1)
    forms = CriticalForms(problem, state, parameter_a, parameter_b)
    q1c = q1.conj()
    rotation = 1j * omega * identity - jacobian
    b00 = forms.state(q0, q0)
    b01 = forms.state(q0, q1)
    b11 = forms.state(q1, q1c)
    g200 = inner(p0, b00).real
    g110 = inner(p1, b01)
    g011 = inner(p0, b11).real
    h200 = -bordered_solve(jacobian, q0, p0, b00 - g200 * q0)
    h020 = np.linalg.solve(2j * omega * identity - jacobian, forms.state(q1, q1))
    h110 = bordered_solve(rotation, q1, p1.conj(), b01 - g110 * q1)
    h011 = -bordered_solve(jacobian, q0, p0, b11 - g011 * q0)
    g300 = inner(p0, forms.state(q0, q0, q0) + 3.0 * forms.state(q0, h200)).real
    g111 = inner(
        p0,
        forms.state(q0, q1, q1c)
        + forms.state(q0, h011)
        + forms.state(q1, h110.conj())
        + forms.state(q1c, h110),
    ).real
    g210 = inner(
        p1,
        forms.state(q0, q0, q1) + 2.0 * forms.state(q0, h110) + forms.state(q1, h200),
    )
    g021 = inner(
        p1,
        forms.state(q1, q1, q1c) + 2.0 * forms.state(q1, h011) + forms.state(q1c, h020),
    )
    return ZeroHopfNormalForm(
        omega,
        q0,
        q1,
        p0,
        p1,
        g200,
        g110,
        g011,
        g300,
        g111,
        g210,
        g021,
    )


class ResonanceError(ValueError):
    """The double Hopf is strongly resonant, so the cubic normal form does not apply."""

    def __init__(self, message: str, ratio: tuple[int, int]) -> None:
        super().__init__(message)
        self.ratio = ratio


def strong_resonance(omega1: float, omega2: float) -> tuple[int, int] | None:
    """The ``omega2 : omega1`` ratio ``(k, l)`` with ``k omega1 = l omega2``, if any.

    Only ``k + l <= 5`` counts: 1:2 means ``omega1 = 2 omega2``.

    Kuznetsov's normal form (3.6) assumes ``k omega1 != l omega2`` for ``k, l > 0``,
    ``k + l <= 5`` (eq. 3.5); otherwise resonant terms of order up to five survive
    and the double Hopf is a different, codim-2 strong resonance.

    Args:
        omega1: The larger frequency.
        omega2: The smaller frequency.

    Returns:
        The first ``(k, l)`` with ``k omega1 = l omega2`` to a relative ``1e-6``,
        or ``None``.

    References:
    - Kuznetsov (1999), eq. (3.5).
    """
    for first in range(1, _RESONANCE_ORDER):
        for second in range(1, _RESONANCE_ORDER + 1 - first):
            if abs(first * omega1 - second * omega2) <= _RESONANCE_TOLERANCE * omega1:
                return first, second
    return None


def _resolvent(jacobian: np.ndarray, frequency: float, right: np.ndarray) -> np.ndarray:
    """``(i frequency I - A)^-1 right``: nonsingular off the critical spectrum."""
    identity = np.eye(jacobian.shape[0])
    return np.linalg.solve(1j * frequency * identity - jacobian, right)


@dataclass(frozen=True)
class HopfHopfNormalForm:
    """The double-Hopf normal form to cubic order (Kuznetsov 1999, eq. 3.6).

    ``w1' = i omega1 w1 + G2100/2 w1|w1|^2 + G1011 w1|w2|^2``,
    ``w2' = i omega2 w2 + G1110 w2|w1|^2 + G0021/2 w2|w2|^2``, with
    ``omega1 > omega2``. In amplitudes ``r_j' = r_j (p_j1 r1^2 + p_j2 r2^2)``.
    """

    omega1: float
    omega2: float
    q1: np.ndarray
    q2: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    g2100: complex
    g1011: complex
    g1110: complex
    g0021: complex

    @property
    def p11(self) -> float:
        """``Re G2100 / 2``: the self-coupling of the first amplitude."""
        return _HALF * self.g2100.real

    @property
    def p12(self) -> float:
        """``Re G1011``: the second amplitude's effect on the first."""
        return self.g1011.real

    @property
    def p21(self) -> float:
        """``Re G1110``: the first amplitude's effect on the second."""
        return self.g1110.real

    @property
    def p22(self) -> float:
        """``Re G0021 / 2``: the self-coupling of the second amplitude."""
        return _HALF * self.g0021.real

    @property
    def theta(self) -> float:
        """``p12 / p22``: the cross coefficient once ``p22`` is scaled to ``-1``."""
        return self.p12 / self.p22

    @property
    def delta(self) -> float:
        """``p21 / p11``: the cross coefficient once ``p11`` is scaled to ``-1``."""
        return self.p21 / self.p11

    @property
    def simple(self) -> bool:
        """The "simple" case ``p11 p22 > 0``; otherwise "difficult"."""
        return self.p11 * self.p22 > 0.0


def hopf_hopf_normal_form(
    problem: TwoParameterProblem,
    state: np.ndarray,
    parameter_a: float,
    parameter_b: float,
) -> HopfHopfNormalForm:
    """Kuznetsov (1999), section 9: eqs. (9.2)-(9.7) and the resonant cubics.

    Args:
        problem: The two-parameter problem.
        state: The double-Hopf equilibrium.
        parameter_a: The first active parameter there.
        parameter_b: The second active parameter there.

    Returns:
        The four resonant cubic coefficients, ordered so ``omega1 > omega2``.

    Raises:
        ResonanceError: If ``k omega1 = l omega2`` with ``k + l <= 5``.

    References:
    - Kuznetsov (1999), eqs. (3.5), (3.6) and (9.2)-(9.7).
    """
    jacobian = problem.state_jacobian(state, parameter_a, parameter_b)
    pairs = _critical_pairs(jacobian)[:2]
    omega1, omega2 = sorted((float(pair.imag) for pair in pairs), reverse=True)
    resonance = strong_resonance(omega1, omega2)
    if resonance is not None:
        message = f"resonant {resonance[0]}:{resonance[1]} double Hopf"
        raise ResonanceError(message, resonance)
    q1 = unit_phase(eigenvector(jacobian, 1j * omega1))
    q2 = unit_phase(eigenvector(jacobian, 1j * omega2))
    p1 = adjoint(jacobian, 1j * omega1, q1)
    p2 = adjoint(jacobian, 1j * omega2, q2)
    forms = CriticalForms(problem, state, parameter_a, parameter_b)
    q1c, q2c = q1.conj(), q2.conj()
    h1100 = _resolvent(jacobian, 0.0, forms.state(q1, q1c))
    h2000 = _resolvent(jacobian, 2.0 * omega1, forms.state(q1, q1))
    h1010 = _resolvent(jacobian, omega1 + omega2, forms.state(q1, q2))
    h1001 = _resolvent(jacobian, omega1 - omega2, forms.state(q1, q2c))
    h0020 = _resolvent(jacobian, 2.0 * omega2, forms.state(q2, q2))
    h0011 = _resolvent(jacobian, 0.0, forms.state(q2, q2c))
    g2100 = inner(
        p1,
        forms.state(q1, q1, q1c)
        + forms.state(h2000, q1c)
        + 2.0 * forms.state(h1100, q1),
    )
    g1011 = inner(
        p1,
        forms.state(q1, q2, q2c)
        + forms.state(h1010, q2c)
        + forms.state(h1001, q2)
        + forms.state(h0011, q1),
    )
    g1110 = inner(
        p2,
        forms.state(q1, q1c, q2)
        + forms.state(h1100, q2)
        + forms.state(h1010, q1c)
        + forms.state(h1001.conj(), q1),
    )
    g0021 = inner(
        p2,
        forms.state(q2, q2, q2c)
        + forms.state(h0020, q2c)
        + 2.0 * forms.state(h0011, q2),
    )
    return HopfHopfNormalForm(
        omega1,
        omega2,
        q1,
        q2,
        p1,
        p2,
        g2100,
        g1011,
        g1110,
        g0021,
    )


@dataclass(frozen=True)
class CuspNormalForm:
    """``w' = c w^3`` on the center manifold (Kuznetsov 1999, section 5)."""

    c: float
    q: np.ndarray
    p: np.ndarray


def cusp_normal_form(
    problem: TwoParameterProblem,
    state: np.ndarray,
    parameter_a: float,
    parameter_b: float,
) -> CuspNormalForm:
    """``c = <p, C(q, q, q) + 3 B(q, h2)> / 6`` (Kuznetsov 1999, section 5).

    Here ``h2 = -A^INV [B(q, q) - <p, B(q, q)> q]``.

    The ``B(q, h2)`` term is the center manifold's curvature; it vanishes only in
    one dimension.

    Args:
        problem: The two-parameter problem.
        state: The cusp equilibrium.
        parameter_a: The first active parameter there.
        parameter_b: The second active parameter there.

    Returns:
        ``c`` and the null vectors; ``c`` does not depend on the sign of ``q``.

    References:
    - Kuznetsov (1999), section 5, eqs. (5.1)-(5.2) and the formula for ``c``.
    """
    jacobian = problem.state_jacobian(state, parameter_a, parameter_b)
    q = _real_unit(right_null(jacobian))
    p = right_null(jacobian.T).real
    p = p / (p @ q)
    forms = CriticalForms(problem, state, parameter_a, parameter_b)
    square = forms.state(q, q).real
    h2 = -bordered_solve(jacobian, q, p, square - (p @ square) * q).real
    cube = forms.state(q, q, q).real + 3.0 * forms.state(q, h2).real
    return CuspNormalForm(float(p @ cube) / 6.0, q, p)


def _sign_word(value: float) -> str:
    return "<0" if value < 0.0 else ">0"


def _classify_bogdanov_takens(
    problem: TwoParameterProblem,
    state: np.ndarray,
    parameter_a: float,
    parameter_b: float,
) -> tuple[str, dict[str, float]]:
    form = bogdanov_takens_normal_form(problem, state, parameter_a, parameter_b)
    label = "degenerate" if form.s == 0 else f"s={form.s:+d}"
    return label, {"a": form.a, "b": form.b, "s": float(form.s)}


def _classify_zero_hopf(
    problem: TwoParameterProblem,
    state: np.ndarray,
    parameter_a: float,
    parameter_b: float,
) -> tuple[str, dict[str, float]]:
    form = zero_hopf_normal_form(problem, state, parameter_a, parameter_b)
    if form.s == 0 or form.d.real == 0.0:
        return "degenerate", {"b": form.b, "c": form.c, "s": 0.0}
    label = f"s={form.s:+d}, theta{_sign_word(form.theta)}"
    if form.torus:
        label += ", torus"
    return label, {
        "b": form.b,
        "c": form.c,
        "s": float(form.s),
        "theta": form.theta,
        "e": form.e,
        "torus": 1.0 if form.torus else 0.0,
    }


def _classify_hopf_hopf(
    problem: TwoParameterProblem,
    state: np.ndarray,
    parameter_a: float,
    parameter_b: float,
) -> tuple[str, dict[str, float]]:
    try:
        form = hopf_hopf_normal_form(problem, state, parameter_a, parameter_b)
    except ResonanceError as error:
        return f"resonant {error.ratio[0]}:{error.ratio[1]}", {}
    coefficients = {
        "p11": form.p11,
        "p12": form.p12,
        "p21": form.p21,
        "p22": form.p22,
    }
    if form.p11 * form.p22 == 0.0:
        return "degenerate", coefficients
    coefficients["theta"] = form.theta
    coefficients["delta"] = form.delta
    return ("simple" if form.simple else "difficult"), coefficients


def _classify_cusp(
    problem: TwoParameterProblem,
    state: np.ndarray,
    parameter_a: float,
    parameter_b: float,
) -> tuple[str, dict[str, float]]:
    form = cusp_normal_form(problem, state, parameter_a, parameter_b)
    return f"c{_sign_word(form.c)}", {"c": form.c}


_CLASSIFIERS = {
    "bogdanov_takens": _classify_bogdanov_takens,
    "zero_hopf": _classify_zero_hopf,
    "hopf_hopf": _classify_hopf_hopf,
    "cusp": _classify_cusp,
}


def classify(
    kind: str,
    problem: TwoParameterProblem,
    state: np.ndarray,
    parameter_a: float,
    parameter_b: float,
) -> tuple[str | None, dict[str, float]]:
    """Label a detected codim-2 point by the signs of its normal-form coefficients.

    ``bogdanov_takens``: ``s = sign(ab)``. ``zero_hopf``: ``s = sign(bc)``, the sign
    of ``theta``, and ``torus`` when ``s theta < 0``. ``hopf_hopf``: ``simple`` when
    ``p11 p22 > 0``, else ``difficult``. ``cusp``: the sign of ``c``. Other kinds,
    and points where the eigenstructure cannot be resolved, get no label.

    Args:
        kind: The detector's kind: ``bogdanov_takens``, ``zero_hopf``,
            ``hopf_hopf`` or ``cusp``.
        problem: The two-parameter problem.
        state: The located equilibrium.
        parameter_a: The first active parameter there.
        parameter_b: The second active parameter there.

    Returns:
        The label (``None`` for other kinds or on numerical failure) and the
        coefficients behind it. A strongly resonant double Hopf is labelled
        ``resonant k:l`` with no coefficients.

    References:
    - Kuznetsov (1999), section 3.
    """
    classifier = _CLASSIFIERS.get(kind)
    if classifier is None:
        return None, {}
    try:
        with np.errstate(divide="raise", invalid="raise"):
            return classifier(
                problem,
                np.asarray(state, dtype=float),
                parameter_a,
                parameter_b,
            )
    except (np.linalg.LinAlgError, FloatingPointError, IndexError, ZeroDivisionError):
        return None, {}
