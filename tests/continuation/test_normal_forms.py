"""Normal-form coefficients at Bogdanov-Takens, zero-Hopf and Hopf-Hopf points.

Acceptance oracles (DEQ-37):

1. A normal form embedded among extra stable directions and hidden behind a random
   similarity transform must come back with its own coefficients, to rounding.
   The coefficients depend on the scale of the critical eigenvectors, which the
   reduction chooses; the test reads that scale back from the eigenvectors the
   reduction reports and checks the coefficients transform exactly as the scale
   says they must. The scale-free invariants are checked directly.
2. The Bogdanov-Takens unfolding predicts the homoclinic curve. In Kuznetsov's
   normal-form coordinates it is ``beta1 = -(6/25) beta2^2 + O(beta2^3)``; the
   continued homoclinic points of the atlas family, mapped into those coordinates,
   must approach it at first order in ``beta2``.
3. The Hopf coefficient equals what the existing center-manifold reduction gives.

Verification against published worked examples and an independent algorithm sits
beside them: Kuznetsov (1999, section 10) for the fold-Hopf in the "new" Lorenz
model, the MatCont tutorial values for the Bykov-Yablonskii-Kim model, and a
Poincare-Dulac normal form computed in eigen-coordinates for systems whose stable
directions feed back into the critical ones.
"""

import math
import warnings
from dataclasses import replace
from unittest import TestCase

import numpy as np
from scipy.optimize import fsolve

from discrecontinual_equations.continuation.center_manifold import (
    EigenvalueSplit,
    TaylorCenterManifold,
)
from discrecontinual_equations.continuation.codim2_builder import (
    Codim2Driver,
    CurveSeed,
)
from discrecontinual_equations.continuation.codim2_config import Codim2Config
from discrecontinual_equations.continuation.codim2_normal_form import (
    bogdanov_takens_normal_form,
    classify,
    hopf_hopf_normal_form,
    hopf_normal_form,
    zero_hopf_normal_form,
)
from discrecontinual_equations.continuation.connecting_orbit import MeshSpec
from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.homoclinic_curve import HomoclinicCurve
from discrecontinual_equations.continuation.normal_form import first_lyapunov
from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.fields import BogdanovTakensField
from tests.continuation.normal_form_reference import (
    coefficient,
    cubic_normal_form,
    fold_hopf_e,
)
from tests.continuation.normal_form_systems import (
    CurvedBogdanovTakens,
    EmbeddedBogdanovTakens,
    EmbeddedHopfHopf,
    EmbeddedZeroHopf,
    StackedBautin,
    StackedHopfHopf,
    build,
    polynomial_field,
    random_transform,
)
from tests.continuation.test_codim2 import BogdanovTakensNormalForm, HopfHopfSystem

_EXACT = 1.0e-9


def _problem(equation: DifferentialEquation) -> TwoParameterProblem:
    return TwoParameterProblem(equation, 0, 1, 0.0, AutomaticDifferentiation())


def _close(actual: complex, expected: complex, tolerance: float = _EXACT) -> bool:
    return abs(actual - expected) <= tolerance * max(1.0, abs(expected))


class TestEmbeddedBogdanovTakens(TestCase):
    """Oracle 1, Bogdanov-Takens: ``(a, b)`` scale with the chosen ``q0``."""

    def test_returns_the_normal_form_coefficients(self):
        for a, b, seed in [(1.0, -1.0, 3), (-0.7, 2.3, 11), (2.5, 0.4, 29)]:
            EmbeddedBogdanovTakens.a = a
            EmbeddedBogdanovTakens.b = b
            EmbeddedBogdanovTakens.transform = random_transform(4, seed)
            problem = _problem(build(EmbeddedBogdanovTakens, 4))
            result = bogdanov_takens_normal_form(problem, np.zeros(4), 0.0, 0.0)
            inverse = np.linalg.inv(EmbeddedBogdanovTakens.transform)
            scale = (inverse @ result.q0)[0]
            assert _close(result.a, scale * a)
            assert _close(result.b, scale * b)
            assert _close(result.b / result.a, b / a)

    def test_unfolding_maps_onto_the_written_parameters(self):
        # The written unfolding is already y' = m1 + m2 y + a x^2 + b x y, so the
        # fold is exactly m1 = 0 and the Hopf exactly m1 = -a m2^2 / b^2.
        EmbeddedBogdanovTakens.a = -0.7
        EmbeddedBogdanovTakens.b = 2.3
        EmbeddedBogdanovTakens.transform = random_transform(4, 11)
        problem = _problem(build(EmbeddedBogdanovTakens, 4))
        unfolding = bogdanov_takens_normal_form(
            problem,
            np.zeros(4),
            0.0,
            0.0,
        ).unfolding
        for beta2 in (0.02, 0.01):
            step = unfolding.side * beta2
            fold = unfolding.fold(step)
            hopf = unfolding.hopf(step)
            assert abs(fold[0]) < 1.0e-12
            assert abs(hopf[0] + (-0.7) * hopf[1] ** 2 / 2.3**2) < 1.0e-12


class TestBogdanovTakensUnfoldingOrder(TestCase):
    """The predicted fold and Hopf curves are right to second order.

    The unfolding is nonlinear in the parameters and moves the center manifold, so
    the exact curves are only known numerically: at each predicted point the exact
    curve is solved for the first parameter at the same second parameter. The
    distance must fall as ``beta2^3``.
    """

    def _exact(self, problem, alpha, condition, guess_state) -> float:
        def residual(unknowns):
            state, first = unknowns[:4], unknowns[4]
            jacobian = problem.state_jacobian(state, first, alpha[1])
            return [*problem.evaluate(state, first, alpha[1]), condition(jacobian)]

        guess = np.concatenate([guess_state, [alpha[0]]])
        solution = fsolve(residual, guess, xtol=1.0e-14)
        assert np.linalg.norm(residual(solution)) < 1.0e-12
        return float(solution[4])

    def test_fold_and_hopf_errors_are_third_order(self):
        CurvedBogdanovTakens.transform = random_transform(4, 41)
        problem = _problem(build(CurvedBogdanovTakens, 4))
        result = bogdanov_takens_normal_form(problem, np.zeros(4), 0.0, 0.0)
        unfolding = result.unfolding
        conditions = {
            "fold": np.linalg.det,
            "hopf": _critical_real_part,
        }
        for name, condition in conditions.items():
            sizes, errors = [], []
            for beta2 in (0.04, 0.02, 0.01, 0.005):
                step = unfolding.side * beta2
                alpha = getattr(unfolding, name)(step)
                # The Hopf equilibrium sits at w0 = -beta2 / b on the manifold.
                guess = result.q0 * (-step / result.b if name == "hopf" else 0.0)
                exact = self._exact(problem, alpha, condition, guess)
                sizes.append(beta2)
                errors.append(abs(alpha[0] - exact))
            order = np.polyfit(np.log(sizes), np.log(errors), 1)[0]
            assert 2.8 < order < 3.2, (name, order, errors)


def _critical_real_part(jacobian: np.ndarray) -> float:
    """``prod_{i<j} (lambda_i + lambda_j)``: zero where a pair sums to zero (Hopf).

    The determinant of the bialternate product ``2 A (.) I``, smooth in ``A``.
    """
    values = np.linalg.eigvals(jacobian)
    size = len(values)
    product = np.prod(
        [values[i] + values[j] for i in range(size) for j in range(i + 1, size)],
    )
    return float(product.real)


class TestIndependentOfDimension(TestCase):
    """Adding stable modes that do not feed back must not change the coefficients.

    Raw coefficients scale with the critical eigenvectors, which a different
    ambient dimension (and a different random ``T``) normalises differently, so the
    check is on what the scale leaves fixed: the second focal value over the fourth
    power of the center basis' conformal scale (``eta4 = 0`` here, which is the only
    case in which ``eta6`` is an invariant at all), and the double-Hopf ``theta``
    and ``delta``.
    """

    def test_second_focal_value_from_three_to_eight_dimensions(self):
        for size in range(3, 9):
            StackedBautin.transform = random_transform(size, 7 * size + 1)
            equation = build(StackedBautin, size)
            problem = _problem(equation)
            origin = np.zeros(size)
            jacobian = problem.state_jacobian(origin, 0.0, 0.0)
            first, second = TaylorCenterManifold().lyapunov_quantities(
                equation.derivative,
                origin,
                jacobian,
                0.0,
            )
            basis = (
                np.linalg.inv(StackedBautin.transform)
                @ EigenvalueSplit()
                .split(
                    jacobian,
                )
                .transform[:, :2]
            )
            scale = abs(np.linalg.det(basis[:2, :2]))
            assert abs(first) < 1.0e-12
            assert _close(second / scale**2, 0.5)
            assert abs(hopf_normal_form(problem, origin, 0.0, 0.0).g21) < 1.0e-12

    def test_double_hopf_invariants_from_five_to_eight_dimensions(self):
        nf = EmbeddedHopfHopf
        for size in range(5, 9):
            StackedHopfHopf.transform = random_transform(size, 3 * size)
            problem = _problem(build(StackedHopfHopf, size))
            result = hopf_hopf_normal_form(problem, np.zeros(size), 0.0, 0.0)
            assert _close(result.theta, nf.g1011.real / (0.5 * nf.g0021.real))
            assert _close(result.delta, nf.g1110.real / (0.5 * nf.g2100.real))


class TestEmbeddedZeroHopf(TestCase):
    """Oracle 1, zero-Hopf: every coefficient, and the scale-free invariants."""

    def test_returns_the_normal_form_coefficients(self):
        nf = EmbeddedZeroHopf
        for seed in (5, 17):
            nf.transform = random_transform(5, seed)
            problem = _problem(build(nf, 5))
            result = zero_hopf_normal_form(problem, np.zeros(5), 0.0, 0.0)
            inverse = np.linalg.inv(nf.transform)
            c = (inverse @ result.q0)[0].real
            kappa = (inverse @ result.q1)[1]
            radial = 4.0 * abs(kappa) ** 2
            assert _close(result.omega, nf.omega)
            assert _close(result.g200, c * nf.g200)
            assert _close(result.g011, radial * nf.g011 / c)
            assert _close(result.g110, c * nf.g110)
            assert _close(result.g300, c * c * nf.g300)
            assert _close(result.g111, radial * nf.g111)
            assert _close(result.g210, c * c * nf.g210)
            assert _close(result.g021, radial * nf.g021)
            # Invariants of the truncated normal form (Kuznetsov 1999, section 3).
            d = nf.g110 - 1j * nf.omega * nf.g300 / (3.0 * nf.g200)
            assert result.s == int(np.sign(0.5 * nf.g200 * nf.g011))
            assert _close(result.theta, d.real / (0.5 * nf.g200))
            assert result.torus == (result.s * result.theta < 0.0)

    def test_e_is_the_coefficient_left_by_the_normalising_transformation(self):
        nf = EmbeddedZeroHopf
        nf.transform = random_transform(5, 5)
        result = zero_hopf_normal_form(_problem(build(nf, 5)), np.zeros(5), 0.0, 0.0)
        values = {
            name: complex(getattr(result, name))
            for name in ("g200", "g011", "g300", "g111", "g110", "g210", "g021")
        }
        for eta in (0.0, 0.4, -1.1):
            terms = fold_hopf_e(values, result.omega, eta)
            assert abs(terms["v3"]) < 1.0e-12
            assert abs(terms["v|z|^2"]) < 1.0e-12
            assert abs(terms["z|z|^2"].real) < 1.0e-12
            assert _close(terms["v^2 z"].real, result.e, 1.0e-12)


class TestEmbeddedHopfHopf(TestCase):
    """Oracle 1, Hopf-Hopf: the four resonant cubic coefficients and the class."""

    def test_returns_the_normal_form_coefficients(self):
        nf = EmbeddedHopfHopf
        for seed in (7, 23):
            nf.transform = random_transform(5, seed)
            problem = _problem(build(nf, 5))
            result = hopf_hopf_normal_form(problem, np.zeros(5), 0.0, 0.0)
            inverse = np.linalg.inv(nf.transform)
            first = 4.0 * abs((inverse @ result.q1)[0]) ** 2
            second = 4.0 * abs((inverse @ result.q2)[2]) ** 2
            assert _close(result.omega1, nf.omega1)
            assert _close(result.omega2, nf.omega2)
            assert _close(result.g2100, first * nf.g2100)
            assert _close(result.g1011, second * nf.g1011)
            assert _close(result.g1110, first * nf.g1110)
            assert _close(result.g0021, second * nf.g0021)
            p11, p22 = 0.5 * nf.g2100.real, 0.5 * nf.g0021.real
            assert _close(result.theta, nf.g1011.real / p22)
            assert _close(result.delta, nf.g1110.real / p11)
            assert result.simple == (p11 * p22 > 0.0)


class TestAgainstPoincareDulac(TestCase):
    """Cubic coefficients with stable feedback, against an independent algorithm."""

    def _reference(self, field: type[DeterministicFunction], basis, eigenvalues):
        size = basis.shape[0]
        return cubic_normal_form(
            build(field, size).derivative,
            np.zeros(size),
            basis,
            eigenvalues,
        )

    def test_zero_hopf_with_feedback(self):
        omega = 1.4
        linear = np.zeros((5, 5))
        linear[1, 2], linear[2, 1] = -omega, omega
        linear[3, 3], linear[4, 4], linear[3, 4] = -1.0, -2.2, 0.5
        for seed in (1, 2, 3):
            field = polynomial_field(linear, seed)
            problem = _problem(build(field, 5))
            result = zero_hopf_normal_form(problem, np.zeros(5), 0.0, 0.0)
            jacobian = problem.state_jacobian(np.zeros(5), 0.0, 0.0)
            values, vectors = np.linalg.eig(jacobian)
            stable = [i for i in range(5) if values[i].real < -0.5]
            basis = np.column_stack(
                [result.q0, result.q1, result.q1.conj(), *vectors[:, stable].T],
            )
            spectrum = np.array([0.0, 1j * omega, -1j * omega, *values[stable]])
            normal = self._reference(field, basis, spectrum)
            zero, rotation = normal[0], normal[1]
            assert _close(result.g200, 2.0 * coefficient(zero, (2, 0, 0, 0, 0)))
            assert _close(result.g011, coefficient(zero, (0, 1, 1, 0, 0)))
            assert _close(result.g110, coefficient(rotation, (1, 1, 0, 0, 0)))
            assert _close(result.g300, 6.0 * coefficient(zero, (3, 0, 0, 0, 0)))
            assert _close(result.g111, coefficient(zero, (1, 1, 1, 0, 0)))
            assert _close(result.g210, 2.0 * coefficient(rotation, (2, 1, 0, 0, 0)))
            assert _close(result.g021, 2.0 * coefficient(rotation, (0, 2, 1, 0, 0)))

    def test_hopf_hopf_with_feedback(self):
        first, second = 2.1, 1.3
        linear = np.zeros((5, 5))
        linear[0, 1], linear[1, 0] = -first, first
        linear[2, 3], linear[3, 2] = -second, second
        linear[4, 4] = -1.5
        for seed in (4, 5, 6):
            field = polynomial_field(linear, seed)
            problem = _problem(build(field, 5))
            result = hopf_hopf_normal_form(problem, np.zeros(5), 0.0, 0.0)
            jacobian = problem.state_jacobian(np.zeros(5), 0.0, 0.0)
            values, vectors = np.linalg.eig(jacobian)
            stable = int(np.argmin(values.real))
            basis = np.column_stack(
                [
                    result.q1,
                    result.q1.conj(),
                    result.q2,
                    result.q2.conj(),
                    vectors[:, stable],
                ],
            )
            spectrum = np.array(
                [1j * first, -1j * first, 1j * second, -1j * second, values[stable]],
            )
            normal = self._reference(field, basis, spectrum)
            one, two = normal[0], normal[2]
            assert _close(result.g2100, 2.0 * coefficient(one, (2, 1, 0, 0, 0)))
            assert _close(result.g1011, coefficient(one, (1, 0, 1, 1, 0)))
            assert _close(result.g1110, coefficient(two, (1, 1, 1, 0, 0)))
            assert _close(result.g0021, 2.0 * coefficient(two, (0, 0, 2, 1, 0)))


class Lorenz84(DeterministicFunction):
    """The "new" Lorenz model with a = 1/4, b = 4 (Kuznetsov 1999, eq. 10.1)."""

    def eval(
        self,
        point: list,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list:
        forcing = self.parameters[0].value
        drive = self.parameters[1].value
        a, b = 0.25, 4.0
        x, y, z = point[0], point[1], point[2]
        return [
            -y * y - z * z - a * x + a * forcing,
            x * y - b * x * z - y + drive,
            b * x * y + x * z - z,
        ]


class TestLorenz84FoldHopf(TestCase):
    """Kuznetsov (1999), section 10: every printed coefficient, to its digits."""

    def test_matches_the_published_values(self):
        root = math.sqrt(145.0)
        forcing = 3907.0 / 2320.0
        drive = 1297.0 / 9280.0 * root
        state = np.array([9.0 / 8.0, -root / 1160.0, 9.0 * root / 290.0])
        equation = build(Lorenz84, 3)
        equation.derivative.parameters[0].value = forcing
        equation.derivative.parameters[1].value = drive
        problem = _problem(equation)
        result = zero_hopf_normal_form(problem, state, forcing, drive)
        # The paper normalises q0[0] = 1 and q1[1] = 1; rescale to that choice.
        c = 1.0 / result.q0[0].real
        k = 1.0 / result.q1[1]
        radial = abs(k) ** 2
        frequency = math.sqrt(27561455.0) / 1160.0
        mixed = root * math.sqrt(190079.0)
        assert _close(result.omega, frequency, 1.0e-10)
        assert _close(c * result.g200, -124102.0 / 190079.0, 1.0e-9)
        assert _close(
            c * result.g110,
            252130.0 / 190079.0 + 1j * 141.0 / 190079.0 * mixed,
            1.0e-9,
        )
        assert _close(radial / c * result.g011, -6915604.0 / 1710711.0, 1.0e-9)
        # G300 is the one printed value that is not (3.4)'s: the w0^3 coefficient of
        # the restricted field is G300/6 (checked independently by Poincare-Dulac
        # below), and the paper's number is a third of that G300.
        printed_g300 = 25602814720.0 / 36130026241.0
        assert _close(c * c * result.g300, 3.0 * printed_g300, 1.0e-9)
        reference = cubic_normal_form(
            equation.derivative,
            state,
            np.column_stack([result.q0, result.q1, result.q1.conj()]),
            np.array([0.0, 1j * result.omega, -1j * result.omega]),
        )
        assert _close(result.g300, 6.0 * coefficient(reference[0], (3, 0, 0)))
        assert _close(radial * result.g111, 9729482240.0 / 325170236169.0, 1.0e-9)
        assert _close(
            c * c * result.g210,
            -36640288960.0 / 36130026241.0
            + 1j * 959598417280.0 / 6867559257863039.0 * mixed,
            1.0e-9,
        )
        assert _close(
            radial * result.g021,
            -4426799104.0 / 325170236169.0
            + 1j * 51158402858528.0 / 26886494494533797685.0 * mixed,
            1.0e-9,
        )
        # The reduced coefficients printed after them; e scales as c^2.
        assert _close(c * result.b, -0.3264484767, 1.0e-9)
        assert _close(radial / c * result.c, -4.042532023, 1.0e-9)
        assert _close(result.d.real / result.g200, -2.031635268, 1.0e-9)
        # The paper's e(0) = 0.2106514716 is its formula at its printed G300; the
        # same formula at the consistent G300 is the e(0) of the transformation.
        stale = replace(result, g300=printed_g300 / (c * c))
        assert _close(c * c * stale.e, 0.2106514716, 1.0e-9)
        assert _close(c * c * result.e, 1.650328846616923, 1.0e-9)
        assert result.s == 1
        assert result.theta < 0.0
        assert result.torus


class Bykov(DeterministicFunction):
    """Bykov-Yablonskii-Kim CO-oxidation model, MatCont tutorial IV."""

    def eval(
        self,
        point: list,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list:
        q2 = self.parameters[0].value
        k = self.parameters[1].value
        q1, q3, q4, q5, q6 = 2.5, 10.0, 0.0675, 1.0, 0.1
        x, y, s = point[0], point[1], point[2]
        z = 1.0 - x - y - s
        return [
            2.0 * q1 * z * z - 2.0 * q5 * x * x - q3 * x * y,
            q2 * z - q6 * y - q3 * x * y,
            q4 * z - k * q4 * s,
        ]


class TestBykovMatcont(TestCase):
    """Kuznetsov and Neirynck, MatCont tutorial IV (2021), section 1.

    MatCont normalises the critical eigenvectors to unit length (and ``q1 . q0 = 0``
    at a Bogdanov-Takens point), which this library does too, so the printed values
    are comparable up to the sign of ``q0``.
    """

    def _locate(self, residual, guess):
        with warnings.catch_warnings():
            # fsolve reports that it cannot beat xtol=1e-14; the residual decides.
            warnings.simplefilter("ignore", RuntimeWarning)
            solution = fsolve(residual, guess, xtol=1.0e-14)
        assert np.linalg.norm(residual(solution)) < 1.0e-13
        return solution

    def test_bogdanov_takens_coefficients(self):
        equation = build(Bykov, 3)
        problem = _problem(equation)

        def residual(unknowns):
            state, q2, k = unknowns[:3], unknowns[3], unknowns[4]
            jacobian = problem.state_jacobian(state, q2, k)
            minors = sum(
                np.linalg.det(np.delete(np.delete(jacobian, i, 0), i, 1))
                for i in range(3)
            )
            return [*problem.evaluate(state, q2, k), np.linalg.det(jacobian), minors]

        point = self._locate(
            residual,
            [0.115909, 0.315467, 0.288437, 1.417628, 0.971397],
        )
        result = bogdanov_takens_normal_form(problem, point[:3], point[3], point[4])
        sign = np.sign(result.a / -8.378442e-02)
        assert abs(sign * result.a - -8.378442e-02) < 5.0e-8
        assert abs(sign * result.b - -2.136280e00) < 5.0e-6

    def test_first_lyapunov_coefficient(self):
        equation = build(Bykov, 3)
        problem = _problem(equation)

        def residual(unknowns):
            state, q2 = unknowns[:3], unknowns[3]
            jacobian = problem.state_jacobian(state, q2, 0.4)
            minors = sum(
                np.linalg.det(np.delete(np.delete(jacobian, i, 0), i, 1))
                for i in range(3)
            )
            hopf = np.trace(jacobian) * minors - np.linalg.det(jacobian)
            return [*problem.evaluate(state, q2, 0.4), hopf]

        point = self._locate(residual, [0.016357, 0.523973, 0.328336, 1.051558])
        result = hopf_normal_form(problem, point[:3], point[3], 0.4)
        # MatCont prints Re G21 / 2 (Kuznetsov 1999, eq. 2.5); the library's l1 is
        # that over omega (the 1998 slides, and normal_form.first_lyapunov).
        # dl1/dQ2 is about 3.3e3 here, and MatCont's Hopf (printed Q2 = 1.051558)
        # sits 5e-7 from the exact one (1.0515575), so agreement is bounded by its
        # location, not by its 7 printed digits: 1e-6 in Q2 is 3.3e-3 in l1.
        assert abs(result.matcont_lyapunov - 1.070259e01) < 2.0e-3
        assert _close(result.first_lyapunov * result.omega, result.matcont_lyapunov)


class CoupledHopf(DeterministicFunction):
    """Planar Hopf with a two-way coupled stable direction (test_center_manifold)."""

    frequency = 1.0

    def eval(
        self,
        point: list,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list:
        w = type(self).frequency
        x, y, z = point[0], point[1], point[2]
        radius = x * x + y * y
        return [
            -w * y - x * radius + z * x,
            w * x - y * radius,
            -2.0 * z + x * x + x * y,
        ]


class TestHopfRegression(TestCase):
    """Oracle 3: the Hopf coefficient agrees with ``TaylorCenterManifold``."""

    def test_matches_the_taylor_center_manifold(self):
        for frequency in (1.0, 2.5):
            CoupledHopf.frequency = frequency
            equation = build(CoupledHopf, 3)
            problem = _problem(equation)
            origin = np.zeros(3)
            jacobian = problem.state_jacobian(origin, 0.0, 0.0)
            focal, _second = TaylorCenterManifold().lyapunov_quantities(
                equation.derivative,
                origin,
                jacobian,
                0.0,
            )
            result = hopf_normal_form(problem, origin, 0.0, 0.0)
            assert _close(result.omega, frequency)
            assert _close(
                result.first_lyapunov,
                first_lyapunov(problem, origin, 0.0, 0.0, frequency),
                1.0e-10,
            )
            # l1 = Re G21 / (2 omega) is already per unit frequency, as the focal
            # value of the time-scaled field is; the 1/4 is the recursion's own
            # normalisation, the ratio test_center_manifold pins.
            assert _close(focal, 0.25 * result.first_lyapunov, 1.0e-9)


class TestBogdanovTakensHomoclinic(TestCase):
    """Oracle 2: the predicted homoclinic curve against the continued one.

    The continued curve carries its own discretisation error, second order in the
    mesh and (in the BT blow-up time) independent of the amplitude, so it shows up
    as a constant offset in ``b1 / b2^2``: 3e-3 at 121 nodes. One Richardson step
    between 121 and 241 nodes removes it (241 and 321 extrapolate to the same four
    digits), and what is left is the asymptotics alone.
    """

    def _deviation(self, equation, unfolding, epsilon: float, nodes: int) -> tuple:
        half = 5.0 * math.sqrt(2.0) / epsilon
        times = np.linspace(-half, half, nodes)
        scaled = epsilon * times / math.sqrt(2.0)
        profile = 1.0 / np.cosh(scaled) ** 2
        orbit = np.column_stack(
            [
                epsilon**2 * (1.0 - 3.0 * profile),
                epsilon**3 * 3.0 * math.sqrt(2.0) * profile * np.tanh(scaled),
            ],
        )
        mesh = MeshSpec(half, nodes - 1, phase_index=1, phase_value=0.0)
        point = HomoclinicCurve(equation, 0, 1, mesh).solve_point(
            (5.0 / 7.0) * epsilon**2,
            orbit,
            np.array([epsilon**2, 0.0]),
            -(epsilon**4),
        )
        assert point is not None
        beta1, beta2 = unfolding.kuznetsov_parameters(
            np.array([point.continuation, point.sweep]),
        )
        return abs(beta2), beta1 / beta2**2 + 6.0 / 25.0

    def test_continued_points_approach_six_over_twenty_five(self):
        equation = build(BogdanovTakensField, 2)
        problem = _problem(equation)
        unfolding = bogdanov_takens_normal_form(
            problem,
            np.zeros(2),
            0.0,
            0.0,
        ).unfolding
        sizes, deviations = [], []
        for epsilon in (0.12, 0.2, 0.28):
            size, coarse = self._deviation(equation, unfolding, epsilon, 121)
            _size, fine = self._deviation(equation, unfolding, epsilon, 241)
            sizes.append(size)
            deviations.append((4.0 * fine - coarse) / 3.0)
        slope, intercept = np.polyfit(sizes, deviations, 1)
        # -6/25 is the limit (measured intercept 7e-6, i.e. 3e-5 relative) ...
        assert abs(intercept) < 1.0e-4
        # ... and the remainder is first order in beta2 (measured 0.99).
        order = np.polyfit(np.log(sizes), np.log(np.abs(deviations)), 1)[0]
        assert 0.9 < order < 1.1
        assert max(abs(np.polyval([slope, intercept], sizes) - deviations)) < 5.0e-5


class ZeroHopfCoupled(DeterministicFunction):
    """A fold in x coupled to a Hopf in (y, z), with |w|^2 feeding the fold."""

    def eval(
        self,
        point: list,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list:
        mu1 = self.parameters[0].value
        mu2 = self.parameters[1].value
        x, y, z = point[0], point[1], point[2]
        radius = y * y + z * z
        return [
            mu1 - x * x + radius,
            mu2 * y - z + 0.5 * x * y - y * radius,
            y + mu2 * z + 0.5 * x * z - z * radius,
        ]


class NonResonantHopfHopf(HopfHopfSystem):
    """The atlas double Hopf with frequencies 1 and 1.7 instead of 1 and 2."""

    second_frequency = 1.7


class TestCodim2PointsAreLabelled(TestCase):
    """Done when: the codim-2 classes are labelled by coefficient sign."""

    def test_zero_hopf_point_carries_its_class(self):
        equation = build(ZeroHopfCoupled, 3)
        equation.derivative.parameters[0].value = 0.25
        seed = CurveSeed(state=[0.5, 0.0, 0.0], parameter_a=-0.25, frequency=1.0)
        config = Codim2Config(
            continuation_parameter_index=1,
            second_parameter_index=0,
            curve="hopf",
            codim2_detectors=["zero_hopf"],
            initial_parameter=0.25,
            direction=-1,
            parameter_lower_bound=-0.1,
            parameter_upper_bound=0.3,
        )
        result = Codim2Driver.run(config, equation, seed)
        points = [p for p in result.points if p.kind == "zero_hopf"]
        assert len(points) == 1
        point = points[0]
        # b = -1, c = +1 (s = -1); Re d = 0.5 so theta = -0.5 and s theta > 0.
        assert point.classification == "s=-1, theta<0"
        assert point.coefficients["s"] == -1.0
        assert point.coefficients["theta"] < 0.0
        assert point.coefficients["torus"] == 0.0

    def test_bogdanov_takens_point_carries_its_class(self):
        equation = build(BogdanovTakensNormalForm, 2)
        equation.derivative.parameters[0].value = -0.25
        equation.derivative.parameters[1].value = -0.5
        seed = CurveSeed(state=[-0.5, 0.0], parameter_a=-0.25, frequency=1.0)
        config = Codim2Config(
            continuation_parameter_index=0,
            second_parameter_index=1,
            curve="hopf",
            codim2_detectors=["bogdanov_takens"],
            initial_parameter=-0.5,
            direction=1,
            parameter_lower_bound=-0.6,
            parameter_upper_bound=0.05,
        )
        result = Codim2Driver.run(config, equation, seed)
        points = [p for p in result.points if p.kind == "bogdanov_takens"]
        assert len(points) == 1
        # y' = ... + x^2 - x y: a = 1, b = -1 for q0 = e1, so ab < 0.
        assert points[0].classification == "s=-1"
        assert points[0].coefficients["s"] == -1.0

    def test_hopf_hopf_point_carries_its_class(self):
        equation = build(NonResonantHopfHopf, 4)
        equation.derivative.parameters[1].value = 0.5
        seed = CurveSeed(state=[0.0, 0.0, 0.0, 0.0], parameter_a=0.0, frequency=1.0)
        config = Codim2Config(
            continuation_parameter_index=0,
            second_parameter_index=1,
            curve="hopf",
            codim2_detectors=["hopf_hopf"],
            initial_parameter=0.5,
            direction=-1,
            parameter_lower_bound=-0.3,
            parameter_upper_bound=0.6,
        )
        result = Codim2Driver.run(config, equation, seed)
        points = [p for p in result.points if p.kind == "hopf_hopf"]
        assert len(points) == 1
        assert points[0].classification == "simple"
        assert points[0].coefficients["p11"] < 0.0
        assert points[0].coefficients["p22"] < 0.0

    def test_strongly_resonant_double_hopf_is_named_not_normalised(self):
        # Frequencies 1 and 2: k omega1 = l omega2 with k + l = 3, outside (3.5).
        problem = _problem(build(HopfHopfSystem, 4))
        label, coefficients = classify("hopf_hopf", problem, np.zeros(4), 0.0, 0.0)
        assert label == "resonant 1:2"
        assert coefficients == {}
