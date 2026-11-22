"""The Levy areas have the law of the Levy area, from the sampler and from a path.

The reference law is Levy's. For two independent Brownian motions over a step
``h`` and ``A = (1/2) int (W_1 dW_2 - W_2 dW_1)`` (the ``1/2`` convention, under
which ``I_(1,2) = (1/2) dW_1 dW_2 + A``), the characteristic function is
``E exp(i u A) = 1 / cosh(u h / 2)`` and, given the increment ``dW``,
``(b / sinh b) exp(|dW|^2 (1 - b coth b) / (2 h))`` with ``b = u h / 2``. From
these: ``E A = 0``; ``Var A = h^2 / 4``; ``Var(A | dW) = h^2 / 12 + h |dW|^2 /
12``, so the ``h^2 / 12`` of the issue (DEQ-36) is the conditional variance at a
zero increment - the area of the Brownian bridge - and not the unconditional
one; ``E A^4 = 5 h^4 / 16``; and the density is ``sech(pi x / h) / h``, with
distribution function ``(2 / pi) arctan(exp(pi x / h))``. ``A`` is uncorrelated
with every increment and with the area of any other pair, by the symmetry
``W_k -> -W_k``.

The sampler (Mrongowius and Rossler 2022) is exact in its first two moments at
any truncation, and its law tends to Levy's as the truncation grows; its
pathwise error is checked separately, against a reference coupled to the same
Fourier coefficients, at the rate of their Theorem 4.4.
"""

import numpy as np
import pytest
from scipy import special, stats

from discrecontinual_equations.solver.stochastic.wiener import (
    BrownianPath,
    GaussianWienerSource,
    LevyAreaSampler,
    WienerIncrements,
)
from tests.solver.coupled_levy_area import coupled_areas, tail

_STEP = 0.25
_DRAWS = 400000


def _levy_cdf(values: np.ndarray, step: float) -> np.ndarray:
    """Levy's distribution function for the area over a step, ``1/2`` convention."""
    return 2.0 / np.pi * np.arctan(np.exp(np.pi * values / step))


def _draw(dimension: int, terms: int | None = None, seed: int = 0):
    generator = np.random.default_rng(seed)
    delta_w = np.sqrt(_STEP) * generator.standard_normal((_DRAWS, dimension))
    sampler = LevyAreaSampler(np.random.default_rng(seed + 1000))
    return delta_w, sampler.sample(delta_w, _STEP, terms=terms)


class TestSamplerLaw:
    """Oracle 1: ``E A = 0``, ``Var A = h^2/4``, uncorrelated with ``dW``."""

    def test_area_matrix_is_antisymmetric_with_a_zero_diagonal(self):
        _, areas = _draw(3)
        assert areas.shape == (_DRAWS, 3, 3)
        assert np.array_equal(areas, -np.swapaxes(areas, 1, 2))
        assert np.all(np.diagonal(areas, axis1=1, axis2=2) == 0.0)

    def test_mean_zero_and_variance_h_squared_over_four(self):
        _, areas = _draw(2)
        area = areas[:, 0, 1]
        standard_error = np.sqrt(_STEP**2 / 4 / _DRAWS)
        assert abs(np.mean(area)) < 4.0 * standard_error
        # Var(A^2) = E A^4 - (E A^2)^2 = h^4 / 4: the relative standard error of
        # the variance estimate is 2 / sqrt(N), 0.3 per cent here.
        assert np.var(area) == pytest.approx(_STEP**2 / 4, rel=0.015)

    def test_uncorrelated_with_the_increments_and_their_product(self):
        delta_w, areas = _draw(2)
        area = areas[:, 0, 1]
        # E[A^2 dW_1^2] = 5 h^3 / 12 from the conditional variance.
        bound = 4.0 * np.sqrt(5.0 * _STEP**3 / 12.0 / _DRAWS)
        assert abs(np.mean(area * delta_w[:, 0])) < bound
        assert abs(np.mean(area * delta_w[:, 1])) < bound
        assert abs(np.mean(area * delta_w[:, 0] * delta_w[:, 1])) < bound * np.sqrt(
            _STEP,
        )

    def test_conditional_variance_is_h_squared_over_twelve_at_zero_increment(self):
        # E[A^2 | dW] = h^2/12 + h (dW_1^2 + dW_2^2) / 12: the regression of A^2
        # on |dW|^2 has intercept h^2 / 12 and slope h / 12. Leaving out the part
        # of the remainder that depends on dW, or giving it the wrong scale, moves
        # the slope; a wrong scale on the dW-free part moves the intercept.
        delta_w, areas = _draw(2, seed=7)
        squared = delta_w[:, 0] ** 2 + delta_w[:, 1] ** 2
        slope, intercept = np.polyfit(squared, areas[:, 0, 1] ** 2, 1)
        assert intercept == pytest.approx(_STEP**2 / 12, rel=0.05)
        assert slope == pytest.approx(_STEP / 12, rel=0.04)

    def test_areas_of_different_pairs_are_uncorrelated(self):
        _, areas = _draw(3, seed=3)
        bound = 5.0 * _STEP**2 / 4 / np.sqrt(_DRAWS)
        assert abs(np.mean(areas[:, 0, 1] * areas[:, 0, 2])) < bound
        assert abs(np.mean(areas[:, 0, 1] * areas[:, 1, 2])) < bound
        assert np.var(areas[:, 1, 2]) == pytest.approx(_STEP**2 / 4, rel=0.015)

    def test_distribution_is_levys_with_many_terms(self):
        _, areas = _draw(2, terms=40, seed=11)
        result = stats.kstest(areas[:, 0, 1], lambda x: _levy_cdf(x, _STEP))
        assert result.pvalue > 1.0e-3, result
        assert np.mean(areas[:, 0, 1] ** 4) == pytest.approx(
            5.0 * _STEP**4 / 16,
            rel=0.03,
        )

    def test_distribution_is_levys_at_the_default_truncation_of_a_small_step(self):
        # The law depends on the truncation only, not on h; at h = 2^-10 with two
        # drivers the order-one rule gives five terms.
        step = 2.0**-10
        assert LevyAreaSampler.terms(step, 2) == 5
        generator = np.random.default_rng(21)
        delta_w = np.sqrt(step) * generator.standard_normal((_DRAWS, 2))
        areas = LevyAreaSampler(np.random.default_rng(22)).sample(delta_w, step)
        result = stats.kstest(areas[:, 0, 1], lambda x: _levy_cdf(x, step))
        assert result.pvalue > 1.0e-3, result


class TestTruncationRule:
    def test_terms_follow_the_order_one_rule(self):
        # n = ceil(sqrt(m / (12 pi^2)) / sqrt(h)): Kastner and Rossler (2022)
        # (67) with gamma = 1 and c1 = 1, i.e. Mrongowius and Rossler (36) with an
        # L2 error of h^(3/2) per step.
        for step, dimension in [(2.0**-4, 2), (2.0**-8, 2), (2.0**-8, 5), (1e-6, 3)]:
            expected = int(
                np.ceil(np.sqrt(dimension / (12 * np.pi**2)) / np.sqrt(step)),
            )
            assert LevyAreaSampler.terms(step, dimension) == expected
        assert LevyAreaSampler.terms(10.0, 2) == 1


class TestSamplerPathwiseError:
    """The pathwise L2 error against a coupled reference falls like ``1 / n``.

    The reference shares the approximation's Fourier coefficients (see
    :mod:`tests.solver.coupled_levy_area`), so the two differ only in the second
    remainder, which is what Mrongowius and Rossler's Theorem 4.4 (i) bounds:
    ``E|I - I^(n)|^2 <= h^2 m sum k^-4 / (4 pi^2 sum k^-2)``, sums over ``k > n``.
    """

    @staticmethod
    def _errors(terms_list: list[int], samples: int) -> dict[int, float]:
        errors = {}
        for terms in terms_list:
            generator = np.random.default_rng(4)
            delta_w = np.sqrt(_STEP) * generator.standard_normal((samples, 2))
            approximation, reference = coupled_areas(
                generator,
                delta_w,
                _STEP,
                terms,
                reference_terms=2048,
            )
            errors[terms] = float(
                np.sqrt(np.mean((approximation - reference)[:, 0, 1] ** 2)),
            )
        return errors

    def test_error_is_within_theorem_four_four_and_falls_like_one_over_n(self):
        terms_list = [2, 4, 8, 16, 32]
        errors = self._errors(terms_list, samples=3000)
        for terms, error in errors.items():
            fourth = float(special.polygamma(3, terms + 1)) / 6.0
            bound = np.sqrt(_STEP**2 * 2 * fourth / (4 * np.pi**2 * tail(terms)))
            assert error < 1.1 * bound, (terms, error, bound)
        order = -np.polyfit(np.log(terms_list), np.log(list(errors.values())), 1)[0]
        assert 0.8 < order < 1.2, errors

    def test_dropping_the_remainders_is_much_worse(self):
        # The truncated Fourier series alone (no Psi terms) errs by
        # sqrt(3 h^2 / (2 pi^2) tail(n)) (Kastner and Rossler (63)): order 1/2 in
        # n, an order of magnitude worse at 32 terms. Pins that both remainders
        # are used, with the right signs: a sign error in either is a pathwise
        # error as large as the remainder itself, invisible to the law tests.
        errors = self._errors([32], samples=2000)
        fourier_only = np.sqrt(3 * _STEP**2 / (2 * np.pi**2) * tail(32))
        assert errors[32] < 0.2 * fourier_only, (errors, fourier_only)


class TestIteratedIntegrals:
    def test_symmetric_part_is_the_product_of_increments(self):
        delta_w, areas = _draw(3)
        increments = WienerIncrements(
            delta_w=delta_w[0],
            delta_z=np.zeros(3),
            levy_area=areas[0],
        )
        integrals = increments.iterated_integrals(_STEP)
        expected = np.outer(delta_w[0], delta_w[0]) - _STEP * np.eye(3)
        assert np.allclose(integrals + integrals.T, expected, atol=1e-15)
        assert np.allclose(0.5 * (integrals - integrals.T), areas[0], atol=1e-15)
        assert integrals[1, 1] == pytest.approx((delta_w[0, 1] ** 2 - _STEP) / 2)

    def test_iterated_integrals_need_the_levy_area(self):
        increments = GaussianWienerSource(seed=1).increments(0.0, _STEP, 2)
        assert increments.levy_area is None
        with pytest.raises(ValueError, match="Levy area"):
            increments.iterated_integrals(_STEP)


class TestGaussianSourceAreas:
    def test_asking_for_areas_leaves_the_increment_stream_unchanged(self):
        plain = GaussianWienerSource(seed=9)
        with_areas = GaussianWienerSource(seed=9)
        for _ in range(5):
            a = plain.increments(0.0, _STEP, 3)
            b = with_areas.increments(0.0, _STEP, 3, levy_area=True)
            assert np.array_equal(a.delta_w, b.delta_w)
            assert np.array_equal(a.delta_z, b.delta_z)
            assert b.levy_area is not None
            assert b.levy_area.shape == (3, 3)

    def test_seeded_areas_repeat_and_have_the_law(self):
        first = GaussianWienerSource(seed=4).increments(0.0, _STEP, 2, levy_area=True)
        second = GaussianWienerSource(seed=4).increments(0.0, _STEP, 2, levy_area=True)
        assert np.array_equal(first.levy_area, second.levy_area)
        source = GaussianWienerSource(seed=5)
        areas = np.array(
            [
                source.increments(0.0, _STEP, 2, levy_area=True).levy_area[0, 1]
                for _ in range(20000)
            ],
        )
        assert np.var(areas) == pytest.approx(_STEP**2 / 4, rel=0.05)

    def test_a_single_channel_has_a_zero_area(self):
        increments = GaussianWienerSource(seed=2).increments(
            0.0,
            _STEP,
            1,
            levy_area=True,
        )
        assert np.array_equal(increments.levy_area, np.zeros((1, 1)))


class TestTruncationRuleOrder:
    def test_rule_gives_global_order_one_asymptotically(self):
        # The global L2 error of sum_n (I_n - I^(n)_n) over [0, 1], which is the
        # whole of Milstein's error on dx = dW_1, dy = x dW_2 (see
        # test_milstein_levy_area.py), is sqrt(1 / h) times the per-step error.
        # With n = terms(h) it is below h (c1 = 1) and of order 1.0 once n is
        # large enough for the error to fall like 1 / n; at n <= 5 it falls more
        # slowly, so step sizes h >= 2^-10 measure about 0.8.
        generator = np.random.default_rng(13)
        errors = {}
        for exponent in [12, 14, 16, 18]:
            step = 2.0**-exponent
            terms = LevyAreaSampler.terms(step, 2)
            delta_w = np.sqrt(step) * generator.standard_normal((1500, 2))
            approximation, reference = coupled_areas(
                generator,
                delta_w,
                step,
                terms,
                reference_terms=16 * terms,
            )
            per_step = np.sqrt(np.mean((approximation - reference)[:, 0, 1] ** 2))
            errors[step] = float(np.sqrt(1.0 / step) * per_step)
            assert errors[step] < step, errors
        order = np.polyfit(np.log(list(errors)), np.log(list(errors.values())), 1)[0]
        assert 0.9 < order < 1.1, errors


def _path_areas(paths: int, step: float, resolution: float, *, sampled: bool):
    areas = np.empty(paths)
    increments = np.empty((paths, 2))
    for index in range(paths):
        path = BrownianPath.sample(0.0, step, resolution, 2, seed=index)
        if not sampled:
            # The same path values, without the areas inside the grid cells: the
            # reading is then the area of the polygon through the grid nodes.
            path = BrownianPath(0.0, resolution, path.values)
        reading = path.increments(0.0, step, 2, levy_area=True)
        areas[index] = reading.levy_area[0, 1]
        increments[index] = reading.delta_w
    return areas, increments


class TestBrownianPathAreas:
    """Oracle 1, path side: the reading converges to the law as the grid refines.

    The polygon through the grid nodes misses the areas inside the cells, which
    are independent with variance ``delta^2 / 4`` each, so its variance is
    ``h^2 / 4 - h delta / 4``. A sampled path also carries the cell areas (drawn
    by the sampler), and its reading has the full ``h^2 / 4`` at any resolution.
    """

    def test_polygon_reading_converges_to_the_law_as_the_grid_refines(self):
        paths = 12000
        measured = {}
        for cells in [2, 8, 32]:
            areas, _ = _path_areas(paths, _STEP, _STEP / cells, sampled=False)
            measured[cells] = np.var(areas)
            expected = _STEP**2 / 4 * (1 - 1 / cells)
            assert measured[cells] == pytest.approx(expected, rel=0.05), measured
        assert measured[2] < measured[8] < measured[32]

    def test_sampled_path_reading_has_the_law_at_a_coarse_grid(self):
        areas, increments = _path_areas(12000, _STEP, _STEP / 2, sampled=True)
        assert np.var(areas) == pytest.approx(_STEP**2 / 4, rel=0.05)
        bound = 4.0 * np.sqrt(5.0 * _STEP**3 / 12.0 / len(areas))
        assert abs(np.mean(areas * increments[:, 0])) < bound

    def test_polygon_reading_is_the_trapezoidal_double_integral(self):
        path = BrownianPath.sample(0.0, 1.0, 0.125, 2, seed=8)
        polygon = BrownianPath(0.0, 0.125, path.values)
        reading = polygon.increments(0.0, 0.5, 2, levy_area=True)
        nodes = np.array([path.value(k * 0.125) for k in range(5)])
        steps = np.diff(nodes, axis=0)
        relative = nodes - nodes[0]
        # I_(1,2) by the trapezoidal rule: sum of mean(W_1 - W_1(0)) dW_2.
        trapezoid = np.sum(0.5 * (relative[:-1, 0] + relative[1:, 0]) * steps[:, 1])
        integrals = reading.iterated_integrals(0.5)
        assert integrals[0, 1] == pytest.approx(trapezoid, abs=1e-14)

    def test_areas_compose_across_steps_by_chens_relation(self):
        path = BrownianPath.sample(0.0, 1.0, 2.0**-6, 3, seed=6)
        whole = path.increments(0.0, 0.5, 3, levy_area=True)
        first = path.increments(0.0, 0.25, 3, levy_area=True)
        second = path.increments(0.25, 0.25, 3, levy_area=True)
        cross = 0.5 * (
            np.outer(first.delta_w, second.delta_w)
            - np.outer(second.delta_w, first.delta_w)
        )
        assert np.allclose(
            whole.levy_area,
            first.levy_area + second.levy_area + cross,
            atol=1e-14,
        )

    def test_increments_without_areas_are_unchanged(self):
        path = BrownianPath.sample(0.0, 1.0, 0.125, 2, seed=4)
        plain = path.increments(0.0, 0.5, 2)
        assert plain.levy_area is None
        with_areas = path.increments(0.0, 0.5, 2, levy_area=True)
        assert np.array_equal(plain.delta_w, with_areas.delta_w)
        assert np.array_equal(plain.delta_z, with_areas.delta_z)
