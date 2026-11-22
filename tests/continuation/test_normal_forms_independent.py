"""Independent checks of the codim-2 normal forms, sharing no code with the library.

The reference here is a Poincare-Dulac reduction written with plain ``numpy``
tensors: the field is given as its exact multilinear forms ``A``, ``B``, ``C`` (so
no jets, polarisation or :class:`.Multivariate` arithmetic is involved), moved into
the eigenbasis the library reports, and normalised to cubic order by removing
every non-resonant quadratic monomial in every component. The resonant cubic
coefficients it leaves on the critical rows are, by definition, Kuznetsov's
``G``-coefficients for that eigenbasis. The systems are dense (every direction
feeds every other, a stable focus included) and their eigenvectors far from
orthogonal.

Beside it: the fold-Hopf ``e(0)`` as derived symbolically (sympy, by an explicit
near-identity transformation and time reparametrisation of (3.4)), the exact
Lorenz-84 ``G300`` from (8.7) in rational arithmetic, and the Bogdanov-Takens
homoclinic prediction against a homoclinic orbit found by shooting.
"""

import itertools
import math
from typing import Any
from unittest import TestCase

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from discrecontinual_equations.continuation.codim2_normal_form import (
    ZeroHopfNormalForm,
    bogdanov_takens_normal_form,
    hopf_hopf_normal_form,
    strong_resonance,
    zero_hopf_normal_form,
)
from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.normal_form import cubic_fold_coefficient
from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.normal_form_systems import build

_TIGHT = 1.0e-9


def _close(actual: complex, expected: complex, tolerance: float = _TIGHT) -> bool:
    return abs(actual - expected) <= tolerance * max(1.0, abs(expected))


def _symmetric2(tensor: np.ndarray) -> np.ndarray:
    return 0.5 * (tensor + tensor.transpose(0, 2, 1))


def _symmetric3(tensor: np.ndarray) -> np.ndarray:
    total = np.zeros_like(tensor)
    for turn in itertools.permutations((1, 2, 3)):
        total = total + tensor.transpose((0, *turn))
    return total / 6.0


class _Tensors(DeterministicFunction):
    """``x' = A x + B(x, x)/2 + C(x, x, x)/6`` with symmetric ``B`` and ``C``."""

    terms: tuple = ((), (), ())

    def eval(
        self,
        point: list,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list:
        linear, quadratic, cubic = type(self).terms
        rows = [0.0 * point[0] for _ in point]
        for i, j, value in linear:
            rows[i] = rows[i] + value * point[j]
        for i, j, k, value in quadratic:
            rows[i] = rows[i] + value * point[j] * point[k]
        for i, j, k, m, value in cubic:
            rows[i] = rows[i] + value * point[j] * point[k] * point[m]
        return rows


def _monomial_weight(indices: tuple[int, ...]) -> float:
    """The number of orderings of a multiset of indices."""
    counts = [indices.count(value) for value in set(indices)]
    return math.factorial(len(indices)) / math.prod(map(math.factorial, counts))


def _tensor_field(
    linear: np.ndarray,
    quadratic: np.ndarray,
    cubic: np.ndarray,
) -> type[_Tensors]:
    size = linear.shape[0]
    terms1 = tuple(
        (i, j, float(linear[i, j])) for i in range(size) for j in range(size)
    )
    terms2 = tuple(
        (i, j, k, 0.5 * _monomial_weight((j, k)) * float(quadratic[i, j, k]))
        for i in range(size)
        for j in range(size)
        for k in range(j, size)
    )
    terms3 = tuple(
        (i, j, k, m, _monomial_weight((j, k, m)) * float(cubic[i, j, k, m]) / 6.0)
        for i in range(size)
        for j in range(size)
        for k in range(j, size)
        for m in range(k, size)
    )
    return type("Tensors", (_Tensors,), {"terms": (terms1, terms2, terms3)})


def _problem(field: type[DeterministicFunction], size: int) -> TwoParameterProblem:
    return TwoParameterProblem(
        build(field, size),
        0,
        1,
        0.0,
        AutomaticDifferentiation(),
    )


def _poincare_dulac(
    tensors: tuple[np.ndarray, np.ndarray, np.ndarray],
    basis: np.ndarray,
    spectrum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Resonant quadratic and full cubic tensors in coordinates ``x = basis y``.

    ``y = z + H(z, z)/2`` removes every quadratic term with a nonzero divisor
    ``lambda_a + lambda_b - lambda_j``; what is left at second order is
    ``R(z, z)/2``, and at third order ``C(z,z,z)/6 + B(z, h(z)) - Dh(z) R(z,z)/2``.
    """
    _, quadratic, cubic = tensors
    inverse = np.linalg.inv(basis)
    b = np.einsum("ji,iab,ak,bl->jkl", inverse, quadratic, basis, basis)
    c = np.einsum("ji,iabc,ak,bl,cm->jklm", inverse, cubic, basis, basis, basis)
    removed = np.zeros_like(b)
    resonant = np.zeros_like(b)
    size = spectrum.size
    for j, k, m in itertools.product(range(size), repeat=3):
        divisor = spectrum[k] + spectrum[m] - spectrum[j]
        if abs(divisor) > 1.0e-8:
            removed[j, k, m] = b[j, k, m] / divisor
        else:
            resonant[j, k, m] = b[j, k, m]
    third = (
        c / 6.0
        + 0.5 * np.einsum("jam,mbc->jabc", b, removed)
        - 0.5 * np.einsum("jab,acd->jbcd", removed, resonant)
    )
    return resonant, _symmetric3(third)


def _coefficient(tensor: np.ndarray, row: int, indices: tuple[int, ...]) -> complex:
    """The coefficient of ``prod z_indices`` in a row of a symmetric tensor."""
    scale = _monomial_weight(indices) / (2.0 if len(indices) == 2 else 1.0)
    return complex(tensor[(row, *indices)] * scale)


def _skewed_transform(size: int, generator: np.random.Generator) -> np.ndarray:
    """A random similarity whose first two columns are nearly parallel."""
    while True:
        transform = generator.normal(size=(size, size))
        transform[:, 1] = transform[:, 0] + 0.3 * transform[:, 1]
        if np.linalg.cond(transform) < 200.0:
            return transform


def _dense_system(block: np.ndarray, seed: int):
    generator = np.random.default_rng(seed)
    size = block.shape[0]
    transform = _skewed_transform(size, generator)
    linear = transform @ block @ np.linalg.inv(transform)
    quadratic = _symmetric2(0.7 * generator.normal(size=(size,) * 3))
    cubic = _symmetric3(0.7 * generator.normal(size=(size,) * 4))
    return linear, quadratic, cubic


def _stable_modes(linear: np.ndarray) -> tuple[list, list]:
    values, vectors = np.linalg.eig(linear)
    stable = [i for i in range(values.size) if values[i].real < -0.1]
    return [vectors[:, i] for i in stable], [values[i] for i in stable]


class TestTensorPoincareDulac(TestCase):
    """Every ``G`` of the zero-Hopf and double Hopf against plain tensor algebra."""

    def test_zero_hopf_coefficients(self):
        for seed in (101, 102, 103):
            omega = 1.3
            block = np.zeros((6, 6))
            block[1, 2], block[2, 1] = -omega, omega
            block[3, 3] = block[4, 4] = -0.6
            block[3, 4], block[4, 3] = -1.7, 1.7
            block[5, 5] = -1.3
            tensors = _dense_system(block, seed)
            problem = _problem(_tensor_field(*tensors), 6)
            result = zero_hopf_normal_form(problem, np.zeros(6), 0.0, 0.0)
            vectors, values = _stable_modes(tensors[0])
            basis = np.column_stack([result.q0, result.q1, result.q1.conj(), *vectors])
            spectrum = np.array([0.0, 1j * omega, -1j * omega, *values])
            second, third = _poincare_dulac(tensors, basis, spectrum)
            expected = {
                "g200": 2.0 * _coefficient(second, 0, (0, 0)),
                "g011": _coefficient(second, 0, (1, 2)),
                "g110": _coefficient(second, 1, (0, 1)),
                "g300": 6.0 * _coefficient(third, 0, (0, 0, 0)),
                "g111": _coefficient(third, 0, (0, 1, 2)),
                "g210": 2.0 * _coefficient(third, 1, (0, 0, 1)),
                "g021": 2.0 * _coefficient(third, 1, (1, 1, 2)),
            }
            for name, value in expected.items():
                assert _close(getattr(result, name), value), (seed, name)

    def test_double_hopf_coefficients(self):
        for seed in (201, 202, 203):
            first, second = 2.4, 1.1
            block = np.zeros((6, 6))
            block[0, 1], block[1, 0] = -first, first
            block[2, 3], block[3, 2] = -second, second
            block[4, 4], block[5, 5], block[4, 5] = -0.9, -2.0, 1.1
            tensors = _dense_system(block, seed)
            problem = _problem(_tensor_field(*tensors), 6)
            result = hopf_hopf_normal_form(problem, np.zeros(6), 0.0, 0.0)
            vectors, values = _stable_modes(tensors[0])
            basis = np.column_stack(
                [result.q1, result.q1.conj(), result.q2, result.q2.conj(), *vectors],
            )
            spectrum = np.array(
                [1j * first, -1j * first, 1j * second, -1j * second, *values],
            )
            _, third = _poincare_dulac(tensors, basis, spectrum)
            expected = {
                "g2100": 2.0 * _coefficient(third, 0, (0, 0, 1)),
                "g1011": _coefficient(third, 0, (0, 2, 3)),
                "g1110": _coefficient(third, 2, (0, 1, 2)),
                "g0021": 2.0 * _coefficient(third, 2, (2, 2, 3)),
            }
            for name, value in expected.items():
                assert _close(getattr(result, name), value), (seed, name)

    def test_cubic_fold_coefficient_at_a_dense_cusp(self):
        # DEQ-83: the swallowtail test function against the reduction. The quadratic
        # tensor is projected so that <p, B(q, q)> = 0 (a cusp); the cubic left on
        # the critical row is then the restricted field's ``c w^3``.
        for seed in (501, 502, 503):
            block = np.diag([0.0, -0.7, -1.3, -2.1])
            block[1, 2], block[2, 1] = 0.9, 0.1
            linear, quadratic, cubic = _dense_system(block, seed)
            values, vectors = np.linalg.eig(linear)
            order = np.argsort(np.abs(values))
            values, vectors = values[order].real, vectors[:, order].real
            q = vectors[:, 0] / np.linalg.norm(vectors[:, 0])
            p = np.linalg.inv(vectors)[0] * np.linalg.norm(vectors[:, 0])
            quadratic = quadratic - np.einsum(
                "i,j,k->ijk",
                q,
                q,
                q,
            ) * (p @ np.einsum("iab,a,b->i", quadratic, q, q))
            tensors = (linear, quadratic, cubic)
            problem = _problem(_tensor_field(*tensors), 4)
            value = cubic_fold_coefficient(problem, np.zeros(4), 0.0, 0.0)
            basis = np.column_stack([q, *vectors[:, 1:].T])
            _, third = _poincare_dulac(tensors, basis, values)
            assert _close(value, _coefficient(third, 0, (0, 0, 0)).real), seed

    def test_zero_hopf_invariants_survive_a_similarity_and_two_stable_modes(self):
        omega = 0.9
        block = np.zeros((5, 5))
        block[1, 2], block[2, 1] = -omega, omega
        block[3, 3], block[4, 4], block[3, 4] = -1.0, -2.5, 0.4
        linear, quadratic, cubic = _dense_system(block, 301)
        result = zero_hopf_normal_form(
            _problem(_tensor_field(linear, quadratic, cubic), 5),
            np.zeros(5),
            0.0,
            0.0,
        )
        # Two more stable modes, driven by everything but feeding nothing back, and
        # a second random similarity on top: the restricted field is unchanged.
        generator = np.random.default_rng(302)
        size = 7
        big_linear = np.zeros((size, size))
        big_linear[:5, :5] = linear
        big_linear[5, 5], big_linear[6, 6], big_linear[5, 6] = -0.8, -1.9, 0.7
        big_quadratic = np.zeros((size,) * 3)
        big_quadratic[:5, :5, :5] = quadratic
        big_quadratic[5:] = _symmetric2(generator.normal(size=(size,) * 3))[5:]
        big_cubic = np.zeros((size,) * 4)
        big_cubic[:5, :5, :5, :5] = cubic
        transform = _skewed_transform(size, generator)
        inverse = np.linalg.inv(transform)
        moved = _tensor_field(
            transform @ big_linear @ inverse,
            np.einsum("ji,iab,ak,bl->jkl", transform, big_quadratic, inverse, inverse),
            np.einsum(
                "ji,iabc,ak,bl,cm->jklm",
                transform,
                big_cubic,
                inverse,
                inverse,
                inverse,
            ),
        )
        other = zero_hopf_normal_form(
            _problem(moved, size),
            np.zeros(size),
            0.0,
            0.0,
        )
        assert other.s == result.s
        assert _close(other.theta, result.theta)
        assert other.torus == result.torus
        # e scales as the square of q0's scale and b as its first power.
        assert _close(other.e / other.b**2, result.e / result.b**2)


def _derived_e(form: ZeroHopfNormalForm) -> float:
    """``e(0)`` as sympy derived it from (3.4), in the restricted field's terms.

    With ``u' = b u^2 + c r + k u^3 + m u r`` and
    ``z' = i omega z + g u z + s u^2 z + t z r`` (``r = |z|^2``), removing
    ``u^3``, ``u r`` and ``z r`` and making the ``u^2 z`` coefficient real (by
    ``dt = (1 + kappa u + l u^2 + mu r) dtau`` and the resonant quadratic changes
    of ``u`` and ``z``) leaves, independently of the free choices,
    ``e = Re s + Re g (m + 2 Re t) / (2 c) - b Re t / c - 3 Re g k / (2 b)``.
    """
    b, c = 0.5 * form.g200, form.g011
    k, m = form.g300 / 6.0, form.g111
    g, s, t = form.g110, 0.5 * form.g210, 0.5 * form.g021
    return (
        s.real
        + g.real * (m + 2.0 * t.real) / (2.0 * c)
        - b * t.real / c
        - 1.5 * g.real * k / b
    )


class TestFoldHopfE(TestCase):
    def test_closed_form_matches_the_symbolic_derivation(self):
        generator = np.random.default_rng(401)
        for _ in range(20):
            real = generator.normal(size=5)
            complex_ = generator.normal(size=3) + 1j * generator.normal(size=3)
            form = ZeroHopfNormalForm(
                omega=1.0 + abs(real[0]),
                q0=np.zeros(1),
                q1=np.zeros(1),
                p0=np.zeros(1),
                p1=np.zeros(1),
                g200=real[1],
                g110=complex_[0],
                g011=real[2],
                g300=real[3],
                g111=real[4],
                g210=complex_[1],
                g021=complex_[2],
            )
            assert _close(form.e, _derived_e(form), 1.0e-12)


class _Lorenz84(DeterministicFunction):
    def eval(
        self,
        point: list,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list:
        forcing: Any = self.parameters[0].value
        drive: Any = self.parameters[1].value
        x, y, z = point
        return [
            -y * y - z * z - 0.25 * x + 0.25 * forcing,
            x * y - 4.0 * x * z - y + drive,
            4.0 * x * y + x * z - z,
        ]


class TestLorenz84G300(TestCase):
    """Kuznetsov (1999), section 10, with (8.7) evaluated in exact arithmetic.

    sympy, with ``q0 = (1, -1007 sqrt145/188065, -5252 sqrt145/188065)``, gives
    ``G300 = 3 <p0, B(q0, h200)> = 76808444160/36130026241``: exactly three times
    the printed ``25602814720/36130026241``. Every other printed ``G`` agrees with
    (8.2)-(8.10), and the printed ``e(0) = 0.2106514716`` is the closed form at the
    printed ``G300``; at the exact one it is ``1.650328846616923``.
    """

    def test_exact_g300_and_e(self):
        root = math.sqrt(145.0)
        forcing, drive = 3907.0 / 2320.0, 1297.0 / 9280.0 * root
        state = np.array([9.0 / 8.0, -root / 1160.0, 9.0 * root / 290.0])
        equation = build(_Lorenz84, 3)
        problem = TwoParameterProblem(
            equation,
            0,
            1,
            0.0,
            AutomaticDifferentiation(),
        )
        result = zero_hopf_normal_form(problem, state, forcing, drive)
        scale = 1.0 / result.q0[0].real
        assert _close(scale**2 * result.g300, 76808444160.0 / 36130026241.0)
        assert _close(scale**2 * result.e, 1.650328846616923)
        assert _close(scale**2 * _derived_e(result), 1.650328846616923)


class _CurvedPlanarBT(DeterministicFunction):
    """A planar BT, not in normal form, with parameters entering nonlinearly."""

    transform = np.array([[1.0, 0.6], [0.3, 1.2]])

    @staticmethod
    def core(z: list, m1: Any, m2: Any) -> list:
        x, y = z
        return [
            y + 0.3 * x * x + 0.2 * x * y + 0.1 * m2 * x,
            m1
            + m2 * y
            + x * x
            + 0.6 * x * y
            - 0.4 * y * y
            + 0.2 * m1 * x
            + 0.3 * m2 * m2
            + 0.25 * x * x * x,
        ]

    def eval(
        self,
        point: list,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list:
        forward = type(self).transform
        inverse = np.linalg.inv(forward)
        z = [
            float(inverse[i, 0]) * point[0] + float(inverse[i, 1]) * point[1]
            for i in (0, 1)
        ]
        row = self.core(z, self.parameters[0].value, self.parameters[1].value)
        return [
            float(forward[i, 0]) * row[0] + float(forward[i, 1]) * row[1]
            for i in (0, 1)
        ]


def _planar(state: np.ndarray, m1: float, m2: float) -> np.ndarray:
    forward = _CurvedPlanarBT.transform
    z = np.linalg.solve(forward, state)
    return forward @ np.array(_CurvedPlanarBT.core(list(z), m1, m2))


def _planar_jacobian(state: np.ndarray, m1: float, m2: float) -> np.ndarray:
    step = 1.0e-7
    columns = [
        (_planar(state + step * e, m1, m2) - _planar(state - step * e, m1, m2))
        / (2.0 * step)
        for e in np.eye(2)
    ]
    return np.column_stack(columns)


def _equilibria(m1: float, m2: float, scale: float) -> list[np.ndarray]:
    """All equilibria within ``6 scale`` of the origin, by bracketing.

    The first row of the core field is linear in ``y``, so ``y`` is eliminated and
    the second row becomes a scalar function of ``x``.
    """

    def ordinate(x: float) -> float:
        return -(0.3 * x * x + 0.1 * m2 * x) / (1.0 + 0.2 * x)

    def residual(x: float) -> float:
        return float(_CurvedPlanarBT.core([x, ordinate(x)], m1, m2)[1])

    grid = np.linspace(-6.0 * scale, 6.0 * scale, 2001)
    values = [residual(x) for x in grid]
    found = []
    for i in range(grid.size - 1):
        if np.sign(values[i]) != np.sign(values[i + 1]):
            x = brentq(residual, grid[i], grid[i + 1], xtol=1.0e-17)
            found.append(_CurvedPlanarBT.transform @ np.array([x, ordinate(x)]))
    return found


def _saddle_and_focus(m1: float, m2: float, scale: float):
    found = _equilibria(m1, m2, scale)
    saddles = [x for x in found if np.linalg.det(_planar_jacobian(x, m1, m2)) < 0]
    others = [x for x in found if np.linalg.det(_planar_jacobian(x, m1, m2)) > 0]
    saddle = saddles[0]
    return saddle, min(others, key=lambda x: np.linalg.norm(x - saddle))


def _splitting(m1: float, m2: float, scale: float) -> float:
    """Unstable minus stable manifold, where each crosses the ray saddle->focus."""
    saddle, focus = _saddle_and_focus(m1, m2, scale)
    values, vectors = np.linalg.eig(_planar_jacobian(saddle, m1, m2))
    ray = focus - saddle
    normal = np.array([-ray[1], ray[0]])
    offset = 1.0e-6 * np.linalg.norm(ray)

    def crossing(_time: float, state: np.ndarray) -> float:
        return float((state - focus) @ normal)

    crossing.terminal = True  # type: ignore[attr-defined]
    hits: list[float] = []
    for index, sense in ((int(np.argmax(values)), 1.0), (int(np.argmin(values)), -1.0)):
        reach = math.nan
        for branch in (1.0, -1.0):
            start = saddle + branch * offset * vectors[:, index].real
            orbit = solve_ivp(
                lambda _t, x, sense=sense: sense * _planar(x, m1, m2),
                (0.0, 1.0e5),
                start,
                method="DOP853",
                rtol=1.0e-12,
                atol=1.0e-14 * scale,
                events=crossing,
            )
            if orbit.t_events[0].size:
                beyond = (orbit.y_events[0][0] - focus) @ ray / (ray @ ray)
                if beyond > 0.0:
                    reach = float(beyond)
        hits.append(reach)
    return hits[0] - hits[1]


class TestBogdanovTakensHomoclinicByShooting(TestCase):
    """The 49/25 homoclinic prediction against loops found by shooting.

    The system is not its own normal form (quadratic terms in both rows, a cubic,
    parameters entering nonlinearly, a skew linear change of coordinates). At the
    predicted parameters the first parameter is re-solved, at the same second one,
    so that the saddle's unstable manifold returns along its stable manifold. The
    prediction is ``O(beta2^2)``-exact, so the gap must fall faster than
    ``beta2^2``; with the ``K02`` term or the 49/25 wrong it would not.
    """

    def test_gap_falls_faster_than_second_order(self):
        problem = _problem(_CurvedPlanarBT, 2)
        unfolding = bogdanov_takens_normal_form(
            problem,
            np.zeros(2),
            0.0,
            0.0,
        ).unfolding
        sizes, gaps = [], []
        for beta2 in (0.1, 0.07, 0.05):
            predicted = unfolding.homoclinic(unfolding.side * beta2)
            second = float(predicted[1])

            def split(first: float, second: float = second, scale: float = beta2):
                return _splitting(first, second, scale)

            width = 0.3 * abs(predicted[0])
            grid = np.linspace(predicted[0] - width, predicted[0] + width, 9)
            signs = np.sign([split(value) for value in grid])
            change = int(np.flatnonzero(signs[:-1] != signs[1:])[0])
            exact = brentq(split, grid[change], grid[change + 1], xtol=1.0e-16)
            sizes.append(beta2)
            gaps.append(abs(exact - predicted[0]))
        order = np.polyfit(np.log(sizes), np.log(gaps), 1)[0]
        assert order > 2.6, (order, gaps)


class TestStrongResonanceOrder(TestCase):
    """(3.5) excludes ``k omega1 = l omega2`` for every ``k + l <= 5``, not just 4."""

    def test_fifth_order_ratios_are_resonant(self):
        assert strong_resonance(1.5, 1.0) == (2, 3)
        assert strong_resonance(4.0, 1.0) == (1, 4)
        assert strong_resonance(3.0, 1.0) == (1, 3)

    def test_sixth_order_ratios_are_not(self):
        assert strong_resonance(5.0, 1.0) is None
        assert strong_resonance(2.5, 1.0) is None
