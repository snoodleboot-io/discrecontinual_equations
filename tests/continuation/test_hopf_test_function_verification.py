"""Independent checks of the Hopf test function against exact references.

These checks do not share code with the implementation or with
``test_hopf_test_function.py``. Two references are used:

* the bialternate product built straight from the *definition* in Guckenheimer,
  Myers & Sturmfels (1997), section 3, ``(A (.) B)_{pq,rs} = (1/2)(|a_pr a_ps; b_qr
  b_qs| + |b_pr b_ps; a_qr a_qs|)`` with ``B = 2I``, not from the entry table (9);
* matrices built with a *known* spectrum, ``A = Q D Q^T`` with ``Q`` orthogonal and
  ``D`` block diagonal (real eigenvalues, ``2 x 2`` rotation blocks, Jordan
  blocks), so the exact sign of ``prod_{i<j} (lambda_i + lambda_j)`` is the parity
  of its negative real factors, counted from ``D`` without any eigenvalue solve.
"""

import math
from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.bialternate import bialternate_test_value
from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.systems import FerroelectricRing
from discrecontinual_equations.variable import Variable


def bialternate_from_definition(matrix: np.ndarray) -> np.ndarray:
    """``2A (.) I`` from the GMS definition of ``A (.) B`` with ``B = 2I``."""
    dimension = matrix.shape[0]
    twice_identity = 2.0 * np.eye(dimension)
    pairs = [(p, q) for p in range(1, dimension) for q in range(p)]
    result = np.zeros((len(pairs), len(pairs)))
    for row, (p, q) in enumerate(pairs):
        for column, (r, s) in enumerate(pairs):
            first = (
                matrix[p, r] * twice_identity[q, s]
                - matrix[p, s] * twice_identity[q, r]
            )
            second = (
                twice_identity[p, r] * matrix[q, s]
                - twice_identity[p, s] * matrix[q, r]
            )
            result[row, column] = 0.5 * (first + second)
    return result


class KnownSpectrum:
    """A real matrix ``Q D Q^T`` whose eigenvalues are fixed by construction."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._blocks: list[np.ndarray] = []
        self._eigenvalues: list[complex] = []

    def real(self, value: float) -> "KnownSpectrum":
        self._blocks.append(np.array([[value]]))
        self._eigenvalues.append(complex(value))
        return self

    def pair(self, real: float, imaginary: float) -> "KnownSpectrum":
        self._blocks.append(np.array([[real, -imaginary], [imaginary, real]]))
        self._eigenvalues += [complex(real, imaginary), complex(real, -imaginary)]
        return self

    def jordan(self, value: float, size: int) -> "KnownSpectrum":
        self._blocks.append(value * np.eye(size) + np.eye(size, k=1))
        self._eigenvalues += [complex(value)] * size
        return self

    def matrix(self) -> np.ndarray:
        dimension = sum(block.shape[0] for block in self._blocks)
        diagonal = np.zeros((dimension, dimension))
        start = 0
        for block in self._blocks:
            size = block.shape[0]
            diagonal[start : start + size, start : start + size] = block
            start += size
        rotation = np.linalg.qr(self._rng.standard_normal((dimension, dimension)))[0]
        return rotation @ diagonal @ rotation.T

    def exact_sign(self) -> float:
        """Sign of ``prod_{i<j} (lambda_i + lambda_j)`` from the known spectrum."""
        eigenvalues = np.array(self._eigenvalues)
        first, second = np.triu_indices(len(eigenvalues), 1)
        sums = eigenvalues[first] + eigenvalues[second]
        real_factors = sums[sums.imag == 0.0].real
        assert np.all(real_factors != 0.0), "exact product is zero"
        return -1.0 if np.count_nonzero(real_factors < 0.0) % 2 else 1.0


class TestMainBialternateSignError(TestCase):
    def test_definition_gives_the_published_three_by_three_product(self):
        # GMS (1997) p. 12 prints 2A (.) I_3 for a general 3 x 3 matrix.
        a = np.arange(1.0, 10.0).reshape(3, 3) ** 1.5
        published = np.array(
            [
                [a[0, 0] + a[1, 1], a[1, 2], -a[0, 2]],
                [a[2, 1], a[0, 0] + a[2, 2], a[0, 1]],
                [-a[2, 0], a[1, 0], a[1, 1] + a[2, 2]],
            ],
        )
        np.testing.assert_allclose(bialternate_from_definition(a), published)

    def test_damped_pair_with_a_zero_eigenvalue_is_not_a_hopf_zero(self):
        # Eigenvalues -1 +- i and 0: sums -2, -1 +- i, product -4. The original
        # entry rule (+a_qr where GMS (9) has -a_qr for s = p) gave det 0 here.
        matrix = np.array([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, -1.0]])
        determinant = np.linalg.det(bialternate_from_definition(matrix))
        assert abs(determinant - (-4.0)) < 1.0e-12
        assert abs(bialternate_test_value(matrix) - (-math.sqrt(2.0))) < 1.0e-12

    def test_real_eigenvalue_through_zero_beside_a_damped_pair_is_not_hopf(self):
        # Eigenvalues -1 +- i and mu: det(2A (.) I) = -2((mu - 1)^2 + 1) < 0 for
        # every mu. The original entry rule changed sign at mu = 0 (a spurious
        # Hopf at a simple zero eigenvalue).
        def family(mu: float) -> np.ndarray:
            return np.array([[-1.0, 0.0, 1.0], [0.0, mu, 0.0], [-1.0, 0.0, -1.0]])

        for mu in (-0.1, -0.01, 0.01, 0.1):
            assert bialternate_test_value(family(mu)) < 0.0
            assert np.linalg.det(bialternate_from_definition(family(mu))) < 0.0


class TestSignAgainstExactSpectra(TestCase):
    def assert_sign(self, spectrum: KnownSpectrum) -> None:
        matrix = spectrum.matrix()
        value = bialternate_test_value(matrix)
        assert math.isfinite(value)
        assert value != 0.0
        assert math.copysign(1.0, value) == spectrum.exact_sign()

    def test_repeated_real_eigenvalues(self):
        rng = np.random.default_rng(11)
        spectrum = KnownSpectrum(rng).real(-1.5).real(-1.5).real(0.7).real(0.7)
        self.assert_sign(spectrum.real(2.0).real(-0.5))

    def test_defective_eigenvalues(self):
        rng = np.random.default_rng(12)
        spectrum = KnownSpectrum(rng).jordan(-0.8, 3).jordan(0.3, 2).pair(0.5, 1.0)
        self.assert_sign(spectrum.real(-2.0))

    def test_complex_pairs_with_equal_and_opposite_real_parts(self):
        # Cross sums of 0.4 +- i and -0.4 +- 2i have zero real part but are not
        # real; equal real parts -0.3 make further non-real cross sums.
        rng = np.random.default_rng(13)
        spectrum = KnownSpectrum(rng).pair(0.4, 1.0).pair(-0.4, 2.0)
        self.assert_sign(spectrum.pair(-0.3, 0.5).pair(-0.3, 3.0).real(1.1))

    def test_zero_eigenvalue(self):
        rng = np.random.default_rng(14)
        spectrum = KnownSpectrum(rng).real(0.0).pair(-0.2, 1.0).real(-1.0)
        self.assert_sign(spectrum.real(3.0))

    def test_dimension_256_with_eigenvalues_from_1e_minus_6_to_1e6(self):
        rng = np.random.default_rng(15)
        spectrum = KnownSpectrum(rng)
        for _ in range(64):
            scale = 10.0 ** rng.uniform(-6.0, 6.0)
            spectrum.pair(scale * rng.standard_normal(), scale * (0.1 + rng.random()))
            spectrum.real(scale * rng.standard_normal())
            spectrum.real(scale * rng.standard_normal())
        self.assert_sign(spectrum)


class TestMagnitude(TestCase):
    def test_magnitude_counts_the_imaginary_part_of_a_cross_sum(self):
        # Eigenvalues 1 +- 2i, -1.05 +- 3i, -5. Real factors 2 and -2.1, so the
        # sign is negative. The cross sum -0.05 - i has a small real part but is
        # not near zero; the smallest |lambda_i + lambda_j| is |-0.05 + i|.
        rng = np.random.default_rng(16)
        matrix = KnownSpectrum(rng).pair(1.0, 2.0).pair(-1.05, 3.0).real(-5.0).matrix()
        expected = -math.hypot(0.05, 1.0)
        assert abs(bialternate_test_value(matrix) - expected) < 1.0e-12


class TestSignChangesAlongFamilies(TestCase):
    def test_brackets_match_the_definition_determinant(self):
        # Sign changes of the test function along A0 + t A1 are those of
        # det(2A (.) I) from the definition, on a grid, for 40 random families.
        rng = np.random.default_rng(17)
        grid = np.linspace(-3.0, 3.0, 61)
        for _ in range(40):
            dimension = int(rng.integers(3, 8))
            base = rng.standard_normal((dimension, dimension))
            slope = rng.standard_normal((dimension, dimension))
            ours, reference = [], []
            for t in grid:
                matrix = base + t * slope
                ours.append(math.copysign(1.0, bialternate_test_value(matrix)))
                sign, _ = np.linalg.slogdet(bialternate_from_definition(matrix))
                reference.append(sign)
            assert ours == reference


class Coupling(Parameter, name="Coupling", abbreviation="lam"):
    pass


class Gain(Parameter, name="Gain", abbreviation="c"):
    pass


class Field(Parameter, name="Field", abbreviation="eps"):
    pass


class Rayleigh(Parameter, name="Rayleigh number", abbreviation="r"):
    pass


class State(Variable, name="State", abbreviation="s"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


LORENZ_SIGMA = 10.0
LORENZ_BETA = 8.0 / 3.0


class Lorenz(DeterministicFunction):
    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        r = float(self.parameters[0].value or 0.0)
        x, y, z = point
        return [
            LORENZ_SIGMA * (y - x),
            r * x - y - x * z,
            x * y - LORENZ_BETA * z,
        ]


def _equation(function_type: type, parameters: list, dimension: int):
    variables: list[Variable] = [State() for _ in range(dimension)]
    derivative = function_type(
        variables=variables,
        parameters=parameters,
        results=[State() for _ in range(dimension)],
        time=None,
    )
    return DifferentialEquation(
        variables=variables,
        time=Time(),
        parameters=parameters,
        derivative=derivative,
    )


def _hopf_points(equation, start, initial, bounds, step):
    lower, upper = bounds
    found = []
    for direction in (1, -1):
        config = ContinuationConfig(
            continuation_parameter_index=0,
            initial_parameter=initial,
            direction=direction,
            parameter_lower_bound=lower,
            parameter_upper_bound=upper,
            initial_step=step,
            maximum_step=2.5 * step,
            maximum_points=4000,
            detectors=["hopf"],
        )
        branch = ContinuerBuilder.build(config, equation).solve(equation, start)
        found += [p for p in branch.special_points if p.kind == "hopf"]
    return found


class TestContinuerLocatesCoupledHopfPoints(TestCase):
    def assert_ring_hopf_points(self, count: int) -> None:
        # Trivial state of x_i' = x_i - x_i^3 - lam x_{i+1}: Jacobian I - lam P with
        # P the cyclic shift, eigenvalues 1 - lam w^k. Hopf at lam = 1/cos(2 pi k/n)
        # with frequency |tan(2 pi k/n)|; no real pair sums to zero (2 != 0).
        parameters = [Coupling(value=0.0), Gain(value=1.0), Field(value=0.0)]
        equation = _equation(FerroelectricRing, parameters, count)
        found = _hopf_points(equation, [0.0] * count, 0.0, (-5.0, 3.5), 0.02)
        angles = [2.0 * math.pi * k / count for k in range(1, (count + 1) // 2)]
        oracles = {1.0 / math.cos(a): abs(math.tan(a)) for a in angles}
        oracles = {lam: w for lam, w in oracles.items() if -5.0 < lam < 3.5}
        located = sorted({round(p.parameter, 6) for p in found})
        assert len(located) == len(oracles), (located, oracles)
        shift = np.roll(np.eye(count), 1, axis=1)
        for point in found:
            distances = {abs(lam - point.parameter): lam for lam in oracles}
            oracle = distances[min(distances)]
            assert abs(point.parameter - oracle) < 1.0e-7
            assert abs(point.frequency - oracles[oracle]) < 1.0e-6
            # Independent eigenvalue check at the located parameter.
            spectrum = np.linalg.eigvals(np.eye(count) - point.parameter * shift)
            oscillatory = spectrum[np.abs(spectrum.imag) > 1.0e-9]
            assert np.min(np.abs(oscillatory.real)) < 1.0e-7

    def test_seven_cell_ring(self):
        self.assert_ring_hopf_points(7)

    def test_forty_cell_ring(self):
        # 780 pairwise sums: det(2A (.) I) is outside the double range here.
        self.assert_ring_hopf_points(40)

    def test_lorenz_subcritical_hopf(self):
        # Hopf of the C+ equilibria at r = sigma (sigma + beta + 3)/(sigma - beta - 1).
        rayleigh = 15.0
        coordinate = math.sqrt(LORENZ_BETA * (rayleigh - 1.0))
        equation = _equation(Lorenz, [Rayleigh(value=rayleigh)], 3)
        start = [coordinate, coordinate, rayleigh - 1.0]
        found = _hopf_points(equation, start, rayleigh, (10.0, 30.0), 0.1)
        sigma, beta = LORENZ_SIGMA, LORENZ_BETA
        oracle = sigma * (sigma + beta + 3.0) / (sigma - beta - 1.0)
        assert len({round(p.parameter, 6) for p in found}) == 1
        assert abs(found[0].parameter - oracle) < 1.0e-6
        # At the Hopf point the characteristic polynomial's pair is +- i w with
        # w^2 = beta (sigma + r).
        assert abs(found[0].frequency - math.sqrt(beta * (sigma + oracle))) < 1.0e-6
