"""The triple-integral sampler of DEQ-66 against references built here.

* **Coefficients.** ``Cbar_(p3,p2,p1)`` by repeated polynomial antiderivatives
  (``numpy.polynomial.legendre.legint``), not by the module's quadrature; and
  Kuznetsov's quoted mean-square error for pairwise different channels at ``q =
  6``, ``0.01956 h^3`` (arXiv:1802.00888, (63)), and the twofold one, (52).
* **Pathwise, the series and its Ito corrections.** The truncated Legendre series
  of the Brownian path is a polynomial curve; its iterated Riemann-Stieltjes
  integrals, by quadrature here, are the plain multiple sums, and Kuznetsov's Ito
  integral (55) is that less the pairings of equal channels at equal degrees,
  which are computed here from the test's own coefficients. With three different
  channels there is no correction and the sampler must return the curve's
  integral itself.
* **The law, jointly with the lower-order integrals.** Moments and covariances of
  every ``I_(j,k,l)`` with ``dW``, ``I_(j,0)``, ``I_(j,k)`` and their products, the
  sampler's against a fine-grid Ito (left-point) Riemann sum of simulated Brownian
  paths. A missing Ito correction shows up here as a correlation of about 0.6
  with one increment, an index in the wrong order as one of 0.5 or more.
"""

import numpy as np
import pytest
from numpy.polynomial import legendre

from discrecontinual_equations.solver.stochastic.iterated_integrals import (
    GaussianIteratedIntegralSource,
    IteratedIntegralSampler,
)
from discrecontinual_equations.solver.stochastic.wiener import GaussianWienerSource


def _cbar_by_antiderivatives(terms: int) -> np.ndarray:
    """``int_{-1}^1 P_p3 int_{-1}^z P_p2 int_{-1}^y P_p1``, as polynomials."""
    bar = np.zeros((terms + 1,) * 3)
    for p1 in range(terms + 1):
        inner = legendre.legint(np.eye(p1 + 1)[p1], lbnd=-1)
        for p2 in range(terms + 1):
            middle = legendre.legint(
                legendre.legmul(np.eye(p2 + 1)[p2], inner), lbnd=-1
            )
            for p3 in range(terms + 1):
                outer = legendre.legint(
                    legendre.legmul(np.eye(p3 + 1)[p3], middle),
                    lbnd=-1,
                )
                bar[p3, p2, p1] = legendre.legval(1.0, outer)
    return bar


def _unit_coefficients(terms: int) -> np.ndarray:
    scale = np.sqrt(2.0 * np.arange(terms + 1) + 1.0)
    return (
        _cbar_by_antiderivatives(terms)
        * np.einsum("a,b,c->abc", scale, scale, scale)
        / 8.0
    )


class TestCoefficients:
    @pytest.mark.parametrize("terms", [0, 1, 3, 7])
    def test_quadrature_matches_polynomial_antiderivatives(self, terms):
        np.testing.assert_allclose(
            IteratedIntegralSampler.coefficients(terms),
            _unit_coefficients(terms),
            atol=1.0e-14,
        )

    def test_kuznetsov_quoted_error_for_three_different_channels(self):
        # arXiv:1802.00888, (63): q = 6 gives "0.01956 (T - t)^3". Computed here
        # 0.0195539 (two ways); the quoted figure agrees to three digits.
        error = IteratedIntegralSampler.triple_errors(6)["distinct"]
        assert error == pytest.approx(0.01956, abs=1.0e-5)

    def test_error_formulas_from_the_test_coefficients(self):
        c = _unit_coefficients(5)
        rest = 1.0 / 6.0 - np.sum(c**2)
        errors = IteratedIntegralSampler.triple_errors(5)
        # (59) j = k != l: C_(p3,p1,p2) C_(p3,p2,p1); (60) j != k = l:
        # C_(p2,p3,p1) C_(p3,p2,p1); (61) j = l != k: C_(p3,p2,p1) C_(p1,p2,p3).
        inner = sum(
            c[a, b, d] * c[a, d, b]
            for a in range(6)
            for b in range(6)
            for d in range(6)
        )
        outer = sum(
            c[b, a, d] * c[a, b, d]
            for a in range(6)
            for b in range(6)
            for d in range(6)
        )
        ends = sum(
            c[a, b, d] * c[d, b, a]
            for a in range(6)
            for b in range(6)
            for d in range(6)
        )
        assert errors["distinct"] == pytest.approx(rest, abs=1.0e-14)
        assert errors["inner"] == pytest.approx(rest - inner, abs=1.0e-14)
        assert errors["outer"] == pytest.approx(rest - outer, abs=1.0e-14)
        assert errors["ends"] == pytest.approx(rest - ends, abs=1.0e-14)

    def test_twofold_error_is_kuznetsov_52(self):
        for q in [1, 4, 30]:
            series = 0.5 - sum(1.0 / (4.0 * i * i - 1.0) for i in range(1, q + 1))
            assert IteratedIntegralSampler.double_error(q) == pytest.approx(
                0.5 * series,
                rel=1.0e-12,
            )


class TestTruncation:
    @pytest.mark.parametrize("step", [2.0**-k for k in range(1, 9)])
    def test_double_terms_is_the_least_meeting_h_to_the_four(self, step):
        q = IteratedIntegralSampler.double_terms(step)
        assert IteratedIntegralSampler.double_error(q) * step**2 <= step**4 or q == 1
        if q > 1:
            assert IteratedIntegralSampler.double_error(q - 1) * step**2 > step**4

    @pytest.mark.parametrize("step", [2.0**-k for k in range(1, 9)])
    def test_triple_terms_is_the_least_meeting_h_to_the_four(self, step):
        q = IteratedIntegralSampler.triple_terms(step)
        worst = max(IteratedIntegralSampler.triple_errors(q).values())
        assert worst * step**3 <= step**4
        if q > 0:
            before = max(IteratedIntegralSampler.triple_errors(q - 1).values())
            assert before * step**3 > step**4

    def test_measured_values(self):
        # The triple error falls like h^3 / (8 q), so q is about 0.13 / h.
        rows = [
            (2.0**-k, IteratedIntegralSampler.triple_terms(2.0**-k))
            for k in range(4, 9)
        ]
        assert [q for _, q in rows] == [2, 4, 8, 16, 33]
        assert IteratedIntegralSampler.double_terms(2.0**-6) == 512


def _legendre_path(
    zeta: np.ndarray, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``W(s)`` and ``W'(s)`` on ``[0, 1]`` for the series with coefficients ``zeta``.

    ``W(s) = sum_p zeta_p int_0^s phi_p``, ``phi_p(s) = sqrt(2p+1) P_p(2s - 1)``.
    """
    terms = zeta.shape[0]
    x = 2.0 * times - 1.0
    value = np.zeros((times.size, zeta.shape[1]))
    slope = np.zeros((times.size, zeta.shape[1]))
    for p in range(terms):
        basis = np.eye(p + 1)[p]
        scale = np.sqrt(2.0 * p + 1.0)
        # d s = d x / 2, so int_0^s phi_p = sqrt(2p+1) / 2 int_{-1}^x P_p.
        antiderivative = legendre.legval(x, legendre.legint(basis, lbnd=-1))
        value += np.outer(scale * 0.5 * antiderivative, zeta[p])
        slope += np.outer(scale * legendre.legval(x, basis), zeta[p])
    return value, slope


def _curve_integral(value, slope, times, channels):
    """``int int int dX_j dX_k dX_l`` along the curve, by cumulative trapezoids."""
    j, k, outer = channels
    ds = np.diff(times)

    def cumulative(f):
        return np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * ds)])

    first = value[:, j] - value[0, j]
    second = cumulative(first * slope[:, k])
    return cumulative(second * slope[:, outer])[-1]


class TestPathwise:
    _TERMS = 6

    def _zeta(self, seed):
        return np.random.default_rng(seed).standard_normal((self._TERMS + 1, 3))

    def test_three_different_channels_are_the_curve_integral(self):
        times = np.linspace(0.0, 1.0, 40001)
        for seed in range(3):
            zeta = self._zeta(seed)
            value, slope = _legendre_path(zeta, times)
            sampled = IteratedIntegralSampler.assemble(zeta, 1.0, 1, self._TERMS)
            for channels in [(0, 1, 2), (2, 0, 1), (1, 2, 0)]:
                assert sampled.triple[channels] == pytest.approx(
                    _curve_integral(value, slope, times, channels),
                    abs=1.0e-7,
                )

    def test_repeated_channels_are_the_curve_integral_less_kuznetsov_pairings(self):
        times = np.linspace(0.0, 1.0, 40001)
        c = _unit_coefficients(self._TERMS)
        for seed in range(3):
            zeta = self._zeta(seed)
            value, slope = _legendre_path(zeta, times)
            sampled = IteratedIntegralSampler.assemble(zeta, 1.0, 1, self._TERMS)
            # sum over equal degrees of the paired channels, the others free.
            inner = np.einsum("cbb,c->", c, zeta[:, 1])  # (0, 0, 1)
            outer = np.einsum("bba,a->", c, zeta[:, 1])  # (1, 0, 0)
            ends = np.einsum("aba,b->", c, zeta[:, 1])  # (0, 1, 0)
            for channels, pairing in [
                ((0, 0, 1), inner),
                ((1, 0, 0), outer),
                ((0, 1, 0), ends),
            ]:
                curve = _curve_integral(value, slope, times, channels)
                assert sampled.triple[channels] == pytest.approx(
                    curve - pairing,
                    abs=1.0e-7,
                )

    def test_one_channel_thrice_is_exact(self):
        zeta = self._zeta(4)
        step = 0.3
        sampled = IteratedIntegralSampler.assemble(zeta, step, 1, self._TERMS)
        dw = np.sqrt(step) * zeta[0]
        for i in range(3):
            assert sampled.triple[i, i, i] == pytest.approx(
                (dw[i] ** 3 - 3.0 * step * dw[i]) / 6.0,
                rel=1.0e-12,
            )

    def test_twofold_curve_integral_for_different_channels(self):
        times = np.linspace(0.0, 1.0, 40001)
        zeta = self._zeta(5)
        value, slope = _legendre_path(zeta, times)
        # The twofold series stops at pairs (i - 1, i), i <= q: it needs zeta_q.
        sampled = IteratedIntegralSampler.assemble(zeta, 1.0, self._TERMS, 0)
        ds = np.diff(times)
        for j, k in [(0, 1), (2, 0)]:
            integrand = (value[:, j] - value[0, j]) * slope[:, k]
            curve = np.sum(0.5 * (integrand[1:] + integrand[:-1]) * ds)
            # The curve's symmetric part is (1/2) X_j(1) X_k(1), the sampler's the
            # Ito (1/2)(dW_j dW_k - delta_jk h); the areas must agree exactly.
            reverse = (value[:, k] - value[0, k]) * slope[:, j]
            curve_reverse = np.sum(0.5 * (reverse[1:] + reverse[:-1]) * ds)
            area = 0.5 * (sampled.double[j, k] - sampled.double[k, j])
            assert area == pytest.approx(0.5 * (curve - curve_reverse), abs=1.0e-8)

    def test_time_integral_is_the_curves_and_reads_back_to_its_coefficients(self):
        # delta_z = int (W(s) - W(0)) ds on the curve, and zeta_0, zeta_1 are
        # recovered from (delta_w, delta_z): the sign of zeta_1 is what ties the
        # triple integrals to the I_(j,0) a source already drew.
        times = np.linspace(0.0, 1.0, 40001)
        zeta = self._zeta(6)
        value, _ = _legendre_path(zeta, times)
        sampled = IteratedIntegralSampler.assemble(zeta, 1.0, 1, 2)
        area = np.trapezoid(value - value[0], times, axis=0)
        np.testing.assert_allclose(sampled.delta_z, area, atol=1.0e-9)
        np.testing.assert_allclose(
            IteratedIntegralSampler.leading(sampled.delta_w, sampled.delta_z, 1.0),
            zeta[:2],
            atol=1.0e-12,
        )
        step = 0.2
        scaled = IteratedIntegralSampler.assemble(zeta, step, 1, 2)
        np.testing.assert_allclose(
            IteratedIntegralSampler.leading(scaled.delta_w, scaled.delta_z, step),
            zeta[:2],
            atol=1.0e-12,
        )

    def test_increments_come_back_unchanged(self):
        source = GaussianWienerSource(3)
        increments = source.increments(0.0, 0.1, 4)
        sampler = IteratedIntegralSampler(np.random.default_rng(0))
        integrals = sampler.complete(increments.delta_w, increments.delta_z, 0.1)
        np.testing.assert_array_equal(integrals.delta_w, increments.delta_w)
        np.testing.assert_array_equal(integrals.delta_z, increments.delta_z)
        # and the symmetric part of the twofold integrals is exact
        dw = increments.delta_w
        np.testing.assert_allclose(
            integrals.double + integrals.double.T,
            np.outer(dw, dw) - 0.1 * np.eye(4),
            atol=1.0e-15,
        )


_CHANNELS = 3
_SAMPLES = 16000
_CHUNK = 2000
_FINE = 256


def _features(dw, dz, double):
    """The lower-order integrals and their products a triple integral meets."""
    n = dw.shape[0]
    cubic = np.einsum("na,nb,nc->nabc", dw, dw, dw).reshape(n, -1)
    mixed = np.einsum("na,nbc->nabc", dw, double).reshape(n, -1)
    timed = np.einsum("na,nb->nab", dz, dw).reshape(n, -1)
    return np.hstack([dw, dz, double.reshape(n, -1), cubic, mixed, timed])


def _riemann(generator, count):
    """Ito (left-point) sums on a 256-cell grid of ``[0, 1]``."""
    cells = generator.standard_normal((count, _FINE, _CHANNELS)) / np.sqrt(_FINE)
    before = np.cumsum(cells, axis=1) - cells
    dz = before.sum(axis=1) / _FINE
    increments = np.einsum("nsj,nsk->nsjk", before, cells)
    double = increments.sum(axis=1)
    double_before = np.cumsum(increments, axis=1) - increments
    triple = np.einsum("nsjk,nsl->njkl", double_before, cells)
    return cells.sum(axis=1), dz, double, triple


def _sampled(generator, count):
    u = generator.standard_normal((count, 2, _CHANNELS))
    dw = u[:, 0]
    dz = 0.5 * (u[:, 0] + u[:, 1] / np.sqrt(3.0))
    integrals = IteratedIntegralSampler(generator).complete(dw, dz, 1.0, terms=(64, 32))
    return dw, dz, integrals.double, integrals.triple


def _moments(draw, seed):
    """Sample means of ``I``, ``I^2``, ``I F`` and ``(I F)^2``, ``F`` the features."""
    generator = np.random.default_rng(seed)
    width = 3 + 3 + 9 + 27 + 27 + 9
    sums = {
        "mean": np.zeros(_CHANNELS**3),
        "square": np.zeros(_CHANNELS**3),
        "fourth": np.zeros(_CHANNELS**3),
        "cross": np.zeros((_CHANNELS**3, width)),
        "cross_square": np.zeros((_CHANNELS**3, width)),
        "feature_square": np.zeros(width),
    }
    for _ in range(_SAMPLES // _CHUNK):
        dw, dz, double, triple = draw(generator, _CHUNK)
        flat = triple.reshape(_CHUNK, -1)
        features = _features(dw, dz, double)
        sums["mean"] += flat.sum(axis=0)
        sums["square"] += np.sum(flat**2, axis=0)
        sums["fourth"] += np.sum(flat**4, axis=0)
        sums["cross"] += flat.T @ features
        sums["cross_square"] += (flat**2).T @ features**2
        sums["feature_square"] += np.sum(features**2, axis=0)
    return {key: value / _SAMPLES for key, value in sums.items()}


@pytest.fixture(scope="module")
def law():
    return _moments(_sampled, 11), _moments(_riemann, 12)


class TestLaw:
    """Studentized: the products here are heavy-tailed (``I_(0,0,1)`` is about
    ``(dW_0^2 - h) dW_1 / 2``), so a fixed tolerance is either loose or flaky."""

    def test_means_are_zero_and_variances_h_cubed_over_six(self, law):
        sampled, _ = law
        assert np.all(np.abs(sampled["mean"]) < 5.0 * np.sqrt(1.0 / 6.0 / _SAMPLES))
        # Ito isometry: E I_(j,k,l)^2 = h^3 / 6 for every pattern of channels,
        # less the truncation's share at q = 32 (Kuznetsov (49), (59)-(61)).
        errors = IteratedIntegralSampler.triple_errors(32)
        expected = np.full((3, 3, 3), 1.0 / 6.0)
        for j, k, n in np.ndindex(3, 3, 3):
            if j == k == n:
                continue
            pattern = (
                "distinct"
                if len({j, k, n}) == 3
                else "inner" if j == k else "outer" if k == n else "ends"
            )
            expected[j, k, n] -= errors[pattern]
        spread = np.sqrt((sampled["fourth"] - sampled["square"] ** 2) / _SAMPLES)
        studentized = (sampled["square"] - expected.ravel()) / spread
        assert np.all(np.abs(studentized) < 5.0), studentized

    def test_known_covariances_with_products_of_increments(self, law):
        sampled, _ = law
        cross = sampled["cross"]
        # E[I_(0,1,2) dW_0 dW_1 dW_2] = h^3 / 6: of the six orderings in the
        # product only (0,1,2) itself is not orthogonal to it.
        cubic = 3 + 3 + 9
        index = np.ravel_multi_index((0, 1, 2), (3, 3, 3))
        assert cross[index, cubic + index] == pytest.approx(1.0 / 6.0, abs=0.02)
        # Ito, not Stratonovich: I_(0,0,1) is uncorrelated with dW_1 (the
        # Stratonovich J_(0,0,1) = I + I_(0,1)/2 has covariance h^2 / 4).
        assert abs(cross[np.ravel_multi_index((0, 0, 1), (3, 3, 3)), 1]) < 0.02

    def test_joint_covariances_match_the_riemann_reference(self, law):
        sampled, reference = law

        def error(moments):
            return (moments["cross_square"] - moments["cross"] ** 2) / _SAMPLES

        difference = sampled["cross"] - reference["cross"]
        studentized = difference / np.sqrt(error(sampled) + error(reference))
        # The Riemann sums carry an O(1/256) bias and the series an O(1/32) one,
        # a few hundredths of the scale; allow that on top of five errors.
        scale = np.sqrt(np.outer(sampled["square"], sampled["feature_square"]))
        allowance = 0.03 * scale / np.sqrt(error(sampled) + error(reference))
        assert np.all(np.abs(studentized) < 5.0 + allowance)
        # and the comparison is not vacuous: the reference has large entries
        assert np.max(np.abs(reference["cross"]) / scale) > 0.5


class TestNestedTruncations:
    def test_mean_square_gap_between_truncations_is_the_error_difference(self):
        # Kuznetsov's truncations are nested projections: E|I^q - I^Q|^2 =
        # e(q) - e(Q) for every pattern. Checks (49) and (59)-(61) by sampling.
        generator = np.random.default_rng(5)
        zeta = generator.standard_normal((6000, 17, 3))
        coarse = IteratedIntegralSampler.assemble(zeta, 1.0, 1, 4).triple
        fine = IteratedIntegralSampler.assemble(zeta, 1.0, 1, 16).triple
        gap = np.mean((coarse - fine) ** 2, axis=0)
        e4, e16 = (IteratedIntegralSampler.triple_errors(q) for q in (4, 16))
        for channels, pattern in [
            ((0, 1, 2), "distinct"),
            ((0, 0, 1), "inner"),
            ((1, 0, 0), "outer"),
            ((0, 1, 0), "ends"),
        ]:
            expected = e4[pattern] - e16[pattern]
            assert gap[channels] == pytest.approx(expected, rel=0.08), pattern


class TestGaussianIteratedIntegralSource:
    def test_increments_are_the_gaussian_sources(self):
        mine, theirs = GaussianIteratedIntegralSource(9), GaussianWienerSource(9)
        for _ in range(3):
            a = mine.increments(0.0, 0.1, 2)
            b = theirs.increments(0.0, 0.1, 2)
            np.testing.assert_array_equal(a.delta_w, b.delta_w)
            np.testing.assert_array_equal(a.delta_z, b.delta_z)

    def test_integrals_draw_the_same_increments_and_repeat_with_the_seed(self):
        first = GaussianIteratedIntegralSource(4).iterated_integrals(0.0, 0.1, 2)
        again = GaussianIteratedIntegralSource(4).iterated_integrals(0.0, 0.1, 2)
        plain = GaussianWienerSource(4).increments(0.0, 0.1, 2)
        np.testing.assert_array_equal(first.delta_w, plain.delta_w)
        np.testing.assert_array_equal(first.triple, again.triple)
        other = GaussianIteratedIntegralSource(5).iterated_integrals(0.0, 0.1, 2)
        assert not np.array_equal(first.triple, other.triple)

    def test_levy_areas_on_request(self):
        increments = GaussianIteratedIntegralSource(1).increments(
            0.0,
            0.1,
            2,
            levy_area=True,
        )
        assert increments.levy_area is not None
