"""The periodic QR algorithm against products whose eigenvalues are known exactly.

Each product is built so its spectrum is fixed by construction, not computed:
factors ``A_k = S_{k+1} D_k S_k^-1`` round a closed chain of similarities
(``S_{p+1} = S_1``) multiply to ``S_1 (D_p ... D_1) S_1^-1``, and permuted
triangular factors with power-of-two diagonals are exact in floating point,
so their product's eigenvalues are exactly the products of the diagonals.
"""

from unittest import TestCase

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from discrecontinual_equations.continuation import periodic_schur
from discrecontinual_equations.continuation.periodic_schur import (
    PeriodicSchurError,
    periodic_eigenvalues,
)


def _worst_relative(computed: np.ndarray, exact: np.ndarray) -> float:
    """Largest relative error, each computed eigenvalue paired with its nearest."""
    gaps = np.abs(computed[:, None] - exact[None, :]) / np.abs(exact)[None, :]
    rows, columns = linear_sum_assignment(gaps)
    return float(np.max(gaps[rows, columns]))


def _product(factors: np.ndarray) -> np.ndarray:
    product = np.eye(factors.shape[1])
    for factor in factors:
        product = factor @ product
    return product


def _similar_chain(
    spectrum: np.ndarray,
    count: int,
    seed: int,
    *,
    orthogonal: bool,
):
    """Factors whose product has ``spectrum`` exactly; each spreads it over ``count``.

    Every factor gets a ``count``-th root of the spectrum with a random spread
    that cancels round the chain, so no factor is the product in disguise.
    """
    rng = np.random.default_rng(seed)
    size = spectrum.size
    if orthogonal:
        frames = [
            np.linalg.qr(rng.standard_normal((size, size)))[0] for _ in range(count)
        ]
    else:
        frames = [
            np.eye(size) + 0.3 * rng.standard_normal((size, size)) for _ in range(count)
        ]
    spread = 0.5 * rng.standard_normal((count, size))
    spread -= spread.mean(axis=0)
    logarithms = np.log(np.abs(spectrum)) / count + spread
    signs = np.sign(spectrum)
    factors = []
    for index in range(count):
        diagonal = np.exp(logarithms[index]) * (signs if index == 0 else 1.0)
        factors.append(
            frames[(index + 1) % count]
            @ np.diag(diagonal)
            @ np.linalg.inv(frames[index]),
        )
    return np.array(factors)


def _permuted_triangular(diagonals: np.ndarray, seed: int) -> np.ndarray:
    """Exact factors: ``P_{k+1} T_k P_k^T`` with power-of-two diagonals.

    A permutation moves entries without rounding them, so these factors are
    exactly what they claim and their product's eigenvalues are exactly the
    products of the diagonals ``T_k`` carry, whatever happens to the rest.
    """
    rng = np.random.default_rng(seed)
    count, size = diagonals.shape
    permutations = [np.eye(size)[rng.permutation(size)] for _ in range(count)]
    factors = []
    for index in range(count):
        triangular = np.triu(rng.integers(-3, 4, (size, size)).astype(float), 1)
        triangular += np.diag(diagonals[index])
        factors.append(
            permutations[(index + 1) % count] @ triangular @ permutations[index].T,
        )
    return np.array(factors)


class TestKnownSpectra(TestCase):
    def test_spectrum_spanning_1e_minus_40_to_1e3(self):
        """Every eigenvalue to its own relative accuracy; the product loses the small.

        The floor is the factors' own: each carries about ``(1e43)^(1/p)`` of
        spread, and rounding a factor costs that times ``eps`` relatively - 4e-8
        at five factors, 1e-15 at 640.
        """
        spectrum = np.array([1.0e3, 1.0, 1.0e-12, 1.0e-40])
        for count, tolerance in ((5, 1e-7), (80, 1e-12), (640, 1e-11)):
            for orthogonal in (True, False):
                factors = _similar_chain(spectrum, count, count, orthogonal=orthogonal)
                computed = periodic_eigenvalues(factors)
                error = _worst_relative(computed, spectrum)
                assert error < tolerance, (count, orthogonal, error)
                explicit = np.linalg.eigvals(_product(factors))
                assert _worst_relative(explicit, spectrum) > 1.0

    def test_exact_factors_with_powers_of_two(self):
        """Each factor halves at least once per position down the diagonal.

        That ordering - an exponential dichotomy, which a cycle's flow has - is
        what makes the eigenvalues well conditioned with respect to the
        factors. Without it random triangular factors make the product's
        eigenvalues so sensitive that perturbing each factor by 1e-15 moved them
        by 6% on 400 factors, which no algorithm working on the factors can
        beat; the check would then measure the problem, not the method.
        """
        rng = np.random.default_rng(11)
        for count, size in ((1, 4), (3, 3), (50, 5), (400, 2)):
            ladder = -2 * np.arange(size)
            exponents = ladder + rng.integers(0, 2, (count, size))
            diagonals = np.ldexp(
                np.where(rng.random((count, size)) < 0.2, -1.0, 1.0),
                exponents,
            )
            factors = _permuted_triangular(diagonals, count)
            exact = np.prod(diagonals, axis=0)
            computed = periodic_eigenvalues(factors)
            assert _worst_relative(computed, exact) < 1e-8, (count, size)

    def test_complex_pairs_and_negative_eigenvalues(self):
        """Blocks ``[[a, -b], [b, a]]`` are ``a + ib``; their products multiply."""
        rng = np.random.default_rng(2)
        count = 120
        frames = [np.linalg.qr(rng.standard_normal((4, 4)))[0] for _ in range(count)]
        exact = np.ones(4, dtype=complex)
        factors = []
        for index in range(count):
            angle, radius = rng.uniform(0.0, 0.2), rng.uniform(0.8, 1.25)
            block = np.zeros((4, 4))
            block[:2, :2] = radius * np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            )
            third, fourth = rng.uniform(0.5, 2.0), rng.uniform(1e-3, 1e-2)
            block[2, 2], block[3, 3] = -third, fourth
            block[:2, 2:] = rng.standard_normal((2, 2))
            exact[0] *= radius * np.exp(1j * angle)
            exact[2] *= -third
            exact[3] *= fourth
            factors.append(frames[(index + 1) % count] @ block @ frames[index].T)
        exact[1] = np.conj(exact[0])
        computed = periodic_eigenvalues(np.array(factors))
        assert _worst_relative(computed, exact) < 1e-10
        # The last eigenvalue is about 1e-276; the explicit product cannot see it.
        assert (
            _worst_relative(np.linalg.eigvals(_product(np.array(factors))), exact) > 1.0
        )

    def test_one_factor_is_the_ordinary_eigenproblem(self):
        rng = np.random.default_rng(4)
        matrix = rng.standard_normal((6, 6))
        computed = periodic_eigenvalues(matrix[None])
        expected = np.linalg.eigvals(matrix)
        assert _worst_relative(computed, expected) < 1e-12


class TestConvergence(TestCase):
    def test_wilkinson_shifts_converge_in_a_few_steps_per_eigenvalue(self):
        """Quadratic convergence: at most four QR steps per eigenvalue.

        Measured two to three on random products; a shift that is not the
        trailing block's nearer eigenvalue converges far more slowly.
        """
        saved = periodic_schur._STEPS_PER_EIGENVALUE  # noqa: SLF001
        periodic_schur._STEPS_PER_EIGENVALUE = 4  # noqa: SLF001
        try:
            rng = np.random.default_rng(8)
            for _ in range(5):
                factors = rng.standard_normal((3, 8, 8))
                computed = periodic_eigenvalues(factors)
                expected = np.linalg.eigvals(_product(factors))
                assert _worst_relative(computed, expected) < 1e-9
        finally:
            periodic_schur._STEPS_PER_EIGENVALUE = saved  # noqa: SLF001


class TestRefusals(TestCase):
    def test_rejects_a_stack_that_is_not_square(self):
        with pytest.raises(ValueError, match="shape"):
            periodic_eigenvalues(np.zeros((3, 2, 4)))

    def test_raises_when_the_iteration_cannot_converge(self):
        """A step budget of zero makes any unreduced product fail loudly."""
        saved = periodic_schur._STEPS_PER_EIGENVALUE  # noqa: SLF001
        periodic_schur._STEPS_PER_EIGENVALUE = 0  # noqa: SLF001
        try:
            with pytest.raises(PeriodicSchurError):
                periodic_eigenvalues(np.array([[[0.0, 1.0], [1.0, 0.0]]] * 3))
        finally:
            periodic_schur._STEPS_PER_EIGENVALUE = saved  # noqa: SLF001
